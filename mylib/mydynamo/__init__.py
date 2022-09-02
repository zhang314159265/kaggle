import torch
from torch import nn, fx
import torchdynamo
from typing import List, Any
from mydynamo import _eval_frame
import os
import re
import dis
import sys
from mylib import misc
import traceback
import itertools
import operator
import inspect
import functools
import dataclasses
import types
import math


SKIP_DIRS = [
    os.path.dirname(__file__) + "/",
]

def skipfiles_check(filename):
    skip_re = re.compile(f"{'|'.join(SKIP_DIRS)}")
    return bool(skip_re.match(filename))

class _MyDynamoContext:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, orig_fn):
        # for nn.Module, optimize the forward method directly
        if isinstance(orig_fn, torch.nn.Module):
            mod = orig_fn
            optimized_forward = self(mod.forward)

            class MyDynamoNNModuleWrapper:
                def __getattr__(self, name):
                    return getattr(mod, name)

                def forward(self, *args, **kwargs):
                    return optimized_forward(*args, **kwargs)

                def __call__(self, *args, **kwargs):
                    return self.forward(*args, **kwargs)

            new_mod = MyDynamoNNModuleWrapper()
            return new_mod

        def fn_wrapper(*args):
            prior = _eval_frame.set_eval_frame(self.callback)
            try:
                res = orig_fn(*args)
            finally:
                _eval_frame.set_eval_frame(prior)
            return res

        return fn_wrapper

class DisableContext(_MyDynamoContext):
    def __init__(self):
        super().__init__(callback=None)

def disable(fn):
    assert fn is not None
    assert callable(fn)
    return DisableContext()(fn)

_unique_id_counter = itertools.count()

@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType

class _NotProvided:
    pass

def create_instruction(name, arg=None, argval=_NotProvided):
    if argval is _NotProvided:
        argval = arg
    return dis.Instruction(
        opname=name,
        opcode=dis.opmap[name],
        arg=arg,
        argval=argval,
        argrepr=None,
        offset=None,
        starts_line=None,
        is_jump_target=None)

def unique_id(name):
    return f"{name}_{next(_unique_id_counter)}"

class Source:
    pass

@dataclasses.dataclass
class LocalSource(Source):
    local_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load(self.local_name)]

    def name(self):
        return self.local_name

class AttrSource(Source):
    base: Source
    member: str

    def __init__(self, base, member):
        super().__init__()
        assert "." not in member
        self.base = base
        self.member = member

    def name(self):
        return f"{self.base.name()}.{self.member}"

# def is_allowed(obj):
#     return obj in {torch, torch.sin, torch.cos}

@functools.lru_cache(None)
def _allowed_function_ids():
    torch_object_ids = dict()

    def _find_torch_objects(module):
        torch_object_ids[id(module)] = module.__name__ 
        # convert items to list to avoid:
        #   RuntimeError: dictionary changed size during iteration
        for name, obj in list(module.__dict__.items()):
            if id(obj) not in torch_object_ids: # to avoid circular reference
                if isinstance(obj, types.ModuleType):
                    if obj.__name__.startswith("torch."):
                        _find_torch_objects(obj)
                elif id(obj) not in torch_object_ids:
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"

    _find_torch_objects(torch)
    _find_torch_objects(math)

    return torch_object_ids

def is_allowed(obj):
    return id(obj) in _allowed_function_ids()

def proxy_args_kwargs(args, kwargs):
    """VariableTracker's to proxy's"""
    proxy_args = tuple(arg.as_proxy() for arg in args)
    proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
    return proxy_args, proxy_kwargs

@dataclasses.dataclass
class GraphArg:
    source: Source
    example: Any

    def load(self, tx):
        # return instructions to load the arg
        return self.source.reconstruct(tx)

    def get_examples(self):
        return [self.example]

class FakeRootModule(torch.nn.Module):
    """Trick the constructor of fx.GraphModule"""
    def __init__(self, nn_modules):
        super().__init__()
        for k, v in nn_modules.items():
            setattr(self, k, v)

class PyCodegen:
    def __init__(self, tx):
        self.tx = tx
        self._output = []
        self.code_options = self.tx.output.code_options
        self.cell_and_freevars = self.tx.cell_and_freevars

    def extend_output(self, insts):
        assert all(isinstance(x, dis.Instruction) for x in insts)
        self._output.extend(insts)

    def append_output(self, inst):
        assert isinstance(inst, dis.Instruction)
        self._output.append(inst)

    def create_load(self, name):
        assert name not in self.cell_and_freevars()
        assert name in self.code_options["co_varnames"], f"{name} missing"
        return create_instruction(
            "LOAD_FAST", self.code_options["co_varnames"].index(name), name
        )

    def create_load_global(self, name, add=False):
        if add:
            self.tx.output.update_co_names(name)
        assert name in self.code_options["co_names"], f"{name} not in co_names"
        return create_instruction("LOAD_GLOBAL", self.code_options["co_names"].index(name), name)
        
    def load_function_name(self, fn_name):
        return [self.create_load_global(fn_name, add=True)]

    def make_call_generated_code(self, fn_name):
        self.extend_output(self.load_function_name(fn_name))

        graphargs = self.tx.output.graphargs
        for arg in graphargs:
            self.extend_output(arg.load(self))

        self.append_output(create_instruction("CALL_FUNCTION", len(graphargs)))

    def get_instructions(self):
        return self._output

class OutputGraph(fx.Tracer):
    def __init__(self, f_globals, code_options, compiler_fn):
        super().__init__()
        self.graph = torch.fx.Graph()
        self.graphargs = []
        self.compiler_fn = compiler_fn
        self.root_globals = f_globals
        self.output_instructions = []
        self.code_options = dict(code_options)
        self.nn_modules = dict()

    def update_co_names(self, name):
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] = tuple(self.code_options["co_names"]) + (name,)

    def add_output_instructions(self, instrs):
        self.output_instructions.extend(instrs)

    def call_user_compiler(self, gm):
        compiled_fn = self.compiler_fn(gm, self.example_inputs())
        assert callable(compiled_fn)
        return compiled_fn

    def example_inputs(self):
        result = []
        for arg in self.graphargs:
            result.extend(arg.get_examples())
        return result

    def install_global(self, name, value):
        # TODO: be able to delete name from root_globals when it's not needed
        self.root_globals[name] = value

    def compile_and_call_fx_graph(self, tx, rv, root):
        assert isinstance(root, FakeRootModule)
        self.create_node("output", "output", (self.create_arg(tuple(x.as_proxy() for x in rv)),), {})

        gm = fx.GraphModule(root, self.graph)
        gm.recompile()
        compiled_fn = self.call_user_compiler(gm)
        compiled_fn = disable(compiled_fn)
        # TODO wrap compiled_fn by disabling dynamo before calling it
        name = unique_id("__compiled_fn")
        self.install_global(name, compiled_fn)

        cg = PyCodegen(tx)
        cg.make_call_generated_code(name)
        return cg.get_instructions()

    def compile_subgraph(self, tx):
        stack_values = list(tx.stack)
        root = FakeRootModule(self.nn_modules)
        # do we really need reverse the stack_values?
        self.add_output_instructions(
            self.compile_and_call_fx_graph(tx, list(reversed(stack_values)), root)
            + [create_instruction("UNPACK_SEQUENCE", len(stack_values))]
        )

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

    def add_submodule(self, mod, *names, **options):
        assert isinstance(mod, torch.nn.Module)
        source = options["source"]

        def wrap_name(module_key):
            return NNModuleVariable(type(mod), module_key, **options)

        for k, v in self.nn_modules.items():
            if v is mod:
                # already exists
                return wrap_name(k)

        # create a new unique name
        name = re.sub(r"[^a-zA-Z0-9]", "_", "_".join(map(str, names)))
        assert name not in self.nn_modules
        self.nn_modules[name] = mod
        return wrap_name(name)

    def get_submodule(self, keys):
        return self.nn_modules[keys]

class VariableBuilder:
    """Wrap a python value in a VariableTracker() instance"""
    def __init__(self, tx, source):
        self.tx = tx
        self.source = source
        self.name = source.name() if source else "SOURCE_SHOULD_NOT_BE_NONE"

    def __call__(self, value):
        return self._wrap(value).clone()

    def get_source(self):
        return self.source

    def wrap_tensor(self, value):
        self.tx.output.graphargs.append(GraphArg(self.get_source(), value))
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
        elif isinstance(value, torch.nn.Module):
            return self.tx.output.add_submodule(
                value,
                self.name,
                source=self.get_source(),
            )
        else:
            assert False, f"_wrap can not handled {value}"

class VariableTracker:
    def __init__(self, source=None):
        self.source = source

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

class NNModuleVariable(VariableTracker):
    def __init__(self, module_type, module_key, **kwargs):
        super().__init__(**kwargs)
        self.module_type = module_type
        self.module_key = module_key
        assert self.source

    def var_getattr(self, tx, name):
        assert self.source
        source = AttrSource(self.source, name)
        base = tx.output.get_submodule(self.module_key)
        base_dict = object.__getattribute__(base, "__dict__")
        if name in base_dict:
            subobj = base_dict[name]
        elif name in base_dict["_modules"]:
            subobj = base_dict["_modules"][name]
        else:
            assert False, "NYI"
        # TODO: why wrap source in NNModuleSource?
        # return VariableBuilder(tx, NNModuleSource(source))(subobj)
        return VariableBuilder(tx, source)(subobj)

    def call_function(self, tx, args, kwargs):
        mod = tx.output.get_submodule(self.module_key)
        assert is_allowed(mod.__class__)
        proxy=tx.output.create_proxy(
            "call_module",
            self.module_key,
            *proxy_args_kwargs(args, kwargs),
        )
        return TensorVariable.create(
            tx=tx,
            proxy=proxy,
        )

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
        if isinstance(obj, NNModuleVariable):
            return obj.var_getattr(tx, name)
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
    def create(cls, tx, example_value=None, proxy=None):
        if example_value is None:
            # TODO: we need create fake tensor and setup example value here
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
    def __init__(self, code_options, instructions, f_globals, f_locals, compiler_fn):
        self.code_options = code_options
        self.instructions = instructions
        self.instruction_pointer = 0
        self.current_instruction = None
        self.f_globals = f_globals
        self.stack = []
        self.output = OutputGraph(f_globals, self.code_options, compiler_fn)
        vars = list(code_options["co_varnames"])
        vars.extend(x for x in self.cell_and_freevars() if x not in vars)
        self.symbolic_locals = {
            k: VariableBuilder(self, LocalSource(k))(f_locals[k]) for k in vars if k in f_locals
        }

    def cell_and_freevars(self):
        if not hasattr(self, "_cell_and_freevars"):
            self._cell_and_freevars = tuple(
                self.code_options["co_cellvars"] or []) + tuple(self.code_options["co_freevars"] or [])
        return self._cell_and_freevars

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
        self.output.add_output_instructions([create_instruction("RETURN_VALUE")])

    BINARY_SUBTRACT = stack_op(operator.sub)

def assemble(instructions, firstlineno):
    lnotab = [] # empty so far
    code = []
    
    for inst in instructions:
        arg = inst.arg or 0
        assert arg >= 0 and arg <= 255
        code.extend((inst.opcode, arg & 0xFF))
    return bytes(code), bytes(lnotab)

def create_callback(compiler_fn):
    def callback(frame):
        """
        Exception thrown from callback will abort _PyEval_EvalFrameDefault since
        the latter assumes no ongoing exceptions.
        """
        code = frame.f_code

        # Fx generated forward method
        # Torchdynamo uses torchdynamo.disable method to skip these calls
        # We should follow the way torchdynamo does so the methods called by
        # the generated forward method will not be translated.
        # if code.co_filename.startswith("<eval_with_key>."):
        #     return

        if skipfiles_check(code.co_filename):
            return None # skip
    
        instructions = list(dis.get_instructions(code))
        print(f"frame: filename {frame.f_code.co_filename}, func name {frame.f_code.co_name}")


        # traceback.print_stack(f=frame)
        for instr in instructions:
            print(f"  {instr}")
    
        try:
            keys = [ # must be a list rather than set to retain order
                "co_argcount",
                "co_posonlyargcount",
                "co_kwonlyargcount",
                "co_nlocals",
                "co_stacksize",
                "co_flags",
                "co_code",
                "co_consts",
                "co_names",
                "co_varnames",
                "co_filename",
                "co_name",
                "co_firstlineno",
                "co_lnotab",
                "co_freevars",
                "co_cellvars",
            ]
            code_options = {k: getattr(code, k) for k in keys}
    
            tracer = InstructionTranslator(
                code_options,
                instructions,
                frame.f_globals,
                frame.f_locals,
                compiler_fn)
    
            tracer.run()
            output = tracer.output
            code_options.update(output.code_options)
            instructions[:] = output.output_instructions
            print("new instruction:");
            for instr in instructions:
                print(f"  {instr}")
    
            bytecode, lnotab = assemble(instructions, code.co_firstlineno)
    
            code_options["co_code"] = bytecode
            code_options["co_nlocals"] = len(code_options["co_varnames"])
            code_options["co_lnotab"] = lnotab
            code = types.CodeType(*[code_options[k] for k in keys]) 
            guarded_code = GuardedCode(code)
            return guarded_code
        except Exception as e:
            print(f"Got exception: {e}")
            traceback.print_exc()
            tracer.dump_stack()
            raise e
    return callback

def optimize(compiler_fn):
    def opt_wrapper(orig_fn):
        return _MyDynamoContext(create_callback(compiler_fn))(orig_fn)
    return opt_wrapper 
