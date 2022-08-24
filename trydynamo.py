import torch
from torch import nn, fx
import torchdynamo
from typing import List
from mydynamo import _eval_frame
import dis
import sys
from mylib import misc
import traceback
import itertools
import operator
import inspect
import functools
import dataclasses

class Source:
    pass

@dataclasses.dataclass
class LocalSource(Source):
    local_name: str

    def name(self):
        return self.local_name

def is_allowed(obj):
    return obj in {torch, torch.sin, torch.cos}

def proxy_args_kwargs(args, kwargs):
    """VariableTracker's to proxy's"""
    proxy_args = tuple(arg.as_proxy() for arg in args)
    proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
    return proxy_args, proxy_kwargs

class GraphArg:
    pass

class OutputGraph(fx.Tracer):
    def __init__(self):
        super().__init__()
        self.graph = torch.fx.Graph()
        self.graphargs = []

    def add_output_instructions(self):
        pass

    def compile_and_call_fx_graph(self, tx, rv):
        self.create_node("output", "output", (self.create_arg(tuple(x.as_proxy() for x in rv)),), {})
        print(f"compile_and_call_fx_graph graph is:")
        self.graph.print_tabular()
        import pdb; pdb.set_trace() # TODO

    def compile_subgraph(self, tx):
        stack_values = list(tx.stack)
        # do we really need reverse the stack_values?
        self.compile_and_call_fx_graph(tx, list(reversed(stack_values)))

    def create_graph_input(self, name):
        placeholders = [n for n in self.graph.nodes if n.op == "placeholder"]

        used_names = {n.target for n in placeholders}
        assert name not in used_names
        if placeholders:
            ctx = self.graph.inserting_after(placeholders[-1])
        else:
            ctx = self.graph.inserting_before(None)  # insert at the beginning of the graph
        with ctx:
            return self.create_proxy("placeholder", name, (), {})

    def create_proxy(
        self,
        op,
        target,
        args,
        kwargs,
        name=None):
        rv = super().create_proxy(op, target, args, kwargs, name)
        return rv

class VariableBuilder:
    """Wrap a python value in a VariableTracker() instance"""
    def __init__(self, tx, source):
        self.tx = tx
        self.source = source
        self.name = source.name() if source else "SOURCE_SHOULD_NOT_BE_NONE"

    def __call__(self, value):
        return self._wrap(value).clone()

    def wrap_tensor(self, value):
        self.tx.output.graphargs.append(GraphArg())
        return TensorVariable.create(
            tx=self.tx,
            proxy=self.tx.output.create_graph_input(self.name),
            example_value=value,
        )

    def _wrap(self, value):
        if isinstance(value, torch.Tensor):
            return self.wrap_tensor(value)
        elif is_allowed(value):
            return TorchVariable(value)
        else:
            assert False, f"_wrap can not handled {value}"

class VariableTracker:
    def clone(self, **kwargs):
        """nop for now"""
        return self

    def call_function(self, tx, args, kwargs):
        raise NotImplementedError(f"call_function {self} {args} {kwargs}")

    def is_python_constant(self):
        try:
            self.as_python_constant()
            return True
        except NotImplementedError:
            return False

class TorchVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def call_function(self, tx, args, kwargs):
        return TensorVariable.create(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                *proxy_args_kwargs(args, kwargs),
            ),
            example_value=None,
        )

class ConstantVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        assert not isinstance(value, torch.Tensor)
        self.value = value

    def as_python_constant(self):
        return self.value

class BuiltinVariable(VariableTracker):
    def __init__(self, fn, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn

    def call_getattr(self, tx, obj, name_var):
        assert name_var.is_python_constant(), "non-const getattr() name"
        name = name_var.as_python_constant()
        if isinstance(obj, TorchVariable):
            member = getattr(obj.value, name)
            assert is_allowed(member), f"{member} is not allowed"
            return TorchVariable(member)
        else:
            assert False, "call_getattr ni" # TODO

    @staticmethod
    @functools.lru_cache(None)
    def _fx_graph_functions():
        fns = {
            operator.add,
            operator.sub,
        }
        return fns

    def can_insert_in_graph(self):
        return self.fn in self._fx_graph_functions()

    def tensor_args(self, *args, **kwargs):
        return any(isinstance(i, TensorVariable) for i in itertools.chain(args, kwargs.values()))

    def call_function(self, tx, args, kwargs):
        tensor_args = self.tensor_args(*args, **kwargs)
        if self.can_insert_in_graph() and tensor_args:
            proxy = tx.output.create_proxy(
                "call_function", self.fn, *proxy_args_kwargs(args, kwargs)
            )
            return TensorVariable.create(
                tx=tx,
                proxy=proxy,
                example_value=None,
            )

        handler = getattr(self, f"call_{self.fn.__name__}", None)

        if handler:
            result = handler(tx, *args, **kwargs)
            assert result is not None
            return result

        return super().call_function(tx, args, kwargs)

class TensorVariable(VariableTracker):
    def __init__(self, proxy, **kwargs):
        super().__init__(**kwargs)
        self.proxy = proxy

    @classmethod
    def create(cls, tx, example_value, proxy=None):
        if example_value  is None:
            # TODO: we need create fake tensor here
            return cls(proxy)

        assert isinstance(example_value, torch.Tensor), f"Got example_value {example_value}"
        return cls(proxy)

    def as_proxy(self):
        return self.proxy

def stack_op(fn):
    nargs = len(inspect.signature(fn).parameters)
    fn_var = BuiltinVariable(fn)

    @functools.wraps(fn)
    def impl(self, inst):
        self.push(fn_var.call_function(self, self.popn(nargs), {}))
    return impl

class InstructionTranslator:
    def __init__(self, code, instructions, f_globals, f_locals):
        self.code = code
        self.instructions = instructions
        self.instruction_pointer = 0
        self.current_instruction = None
        self.f_globals = f_globals
        self.stack = []
        self.output = OutputGraph()
        self.symbolic_locals = {
            k: VariableBuilder(self, LocalSource(k))(f_locals[k]) for k in self.code.co_varnames if k in f_locals
        }

    def call_function(self, fn, args, kwargs):
        assert isinstance(fn, VariableTracker)
        assert isinstance(args, list)
        assert isinstance(kwargs, dict)
        assert all(isinstance(x, VariableTracker) for x in itertools.chain(args, kwargs.values()))
        self.push(fn.call_function(self, args, kwargs))

    def dump_stack(self):
        print(f"Stack size {len(self.stack)}")
        for item in self.stack:
            print(f"  {item}")

    def step(self):
        """Process one instruction. Return false to exit"""
        inst = self.instructions[self.instruction_pointer]
        self.current_instruction = inst
        self.instruction_pointer += 1
        if self.instruction_pointer < len(self.instructions):
            pass
        else:
            self.instruction_pointer = None
        if not hasattr(self, inst.opname):
            raise NotImplementedError(f"missing: {inst.opname}")
        getattr(self, inst.opname)(inst)
        return inst.opname != "RETURN_VALUE"

    def run(self):
        while self.instruction_pointer is not None and self.step():
            pass

    def push(self, val):
        assert val is None or isinstance(val, VariableTracker)
        self.stack.append(val)

    def pop(self):
        return self.stack.pop()

    def popn(self, n):
        assert n >= 0
        return list(reversed([self.pop() for _ in range(n)]))

    def LOAD_ATTR(self, inst):
        obj = self.pop()
        result = BuiltinVariable(getattr).call_function(
            self, [obj, ConstantVariable(inst.argval)], {}
        )
        self.push(result)

    def LOAD_METHOD(self, inst):
        self.LOAD_ATTR(inst)
        self.push(None)

    def LOAD_GLOBAL(self, inst):
        name = inst.argval
        value = self.f_globals[name]
        self.push(VariableBuilder(self, source=None)(value))

    def LOAD_FAST(self, inst):
        name = inst.argval
        self.push(self.symbolic_locals[name])

    def STORE_FAST(self, inst):
        self.symbolic_locals[inst.argval] = self.pop()

    def CALL_METHOD(self, inst):
        args = self.popn(inst.argval)
        dummy = self.pop()
        assert dummy is None
        fn = self.pop()
        self.call_function(fn, args, {})

    def RETURN_VALUE(self, inst):
        self.instruction_poiner = None
        self.output.compile_subgraph(self)
        self.output.add_output_instructions()

    BINARY_SUBTRACT = stack_op(operator.sub)
        
def orig_fn(x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    # why calling misc.sub cause sin/cos disappearing from the graph?
    # res = misc.sub(a, b)
    res = a - b
    return res

""" Graph generated by torchdynamo:
opcode         name    target                                                  args        kwargs
-------------  ------  ------------------------------------------------------  ----------  --------
placeholder    x       x                                                       ()          {}
placeholder    y       y                                                       ()          {}
call_function  cos     <built-in method cos of type object at 0x7f8672191fc0>  (x,)        {}
call_function  sin     <built-in method sin of type object at 0x7f8672191fc0>  (y,)        {}
call_function  sub     <built-in function sub>                                 (cos, sin)  {}
output         output  output                                                  ((sub,),)   {}
"""
def try_dynamo():
    print("Try torchdynamo")
    def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        print("my_compiler() called with FX graph:")
        gm.graph.print_tabular()
        return gm.forward
    
    fn = torchdynamo.optimize(my_compiler)(orig_fn)
    
    fn(torch.randn(10), torch.randn(10))

def try_mydynamo():
    print("Try mydynamo")

    def callback(frame):
        """
        Exception thrown from callback will abort _PyEval_EvalFrameDefault since
        the latter assumes no ongoing exceptions.
        """
        code = frame.f_code
        instructions = list(dis.get_instructions(code))
        print(f"frame is {frame.f_code.co_filename}")
        for instr in instructions:
            print(f"  {instr}")

        try:
            tracer = InstructionTranslator(
                code,
                instructions,
                frame.f_globals,
                frame.f_locals)

            tracer.run()
        except Exception as e:
            print(f"Got exception {e}")
            traceback.print_exc()
            tracer.dump_stack()
            raise e
    
    _eval_frame.set_eval_frame(callback)
    res = orig_fn(torch.randn(10), torch.randn(10))
    _eval_frame.set_eval_frame(None)
    print(f"Result is {res}")

[try_dynamo, try_mydynamo][
1
]()
