import torch

from torch.utils._pytree import tree_map

_orig_module_call = torch.nn.Module.__call__

class Proxy:
    def __init__(self, node):
        self.node = node

class Node:
    def __init__(self, name, node_type, target, args, kwargs):
        self.name = name  # unique name of value being created
        self.node_type = node_type
        self.target = target
        self._prev = self._next = self
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return self.name

    def prepend(self, x):
        x._remove_from_list()
        p = self._prev
        # insert x between p and self
        p._next, x._prev = x, p
        x._next, self._prev = self, x

    def _remove_from_list(self):
        p, n = self._prev, self._next
        p._next, n._prev = n, p

    def format_node(self, placeholder_names):
        def _format_arg(xargs):
            return str(tree_map(lambda x: f"%{x}", xargs))
        if self.node_type == "placeholder":
            if placeholder_names is not None:
                placeholder_names.append(self.target)
                return None
            assert False, "Only support cases with placeholder_names right now"
        elif self.node_type == "output":
            return f"return {self.args[0]}"
        elif self.node_type == "call_module":
            return f"%{self.name} = {self.node_type}[target={self.target}](args = {_format_arg(self.args)}, kwargs = {_format_arg(self.kwargs)})"
        else:
            raise RuntimeError(f"Unexpected node type: {self.node_type}")

class _node_list:
    def __init__(self, graph, direction = "_next"):
        assert direction in ["_next", "_prev"]
        self.graph = graph
        self.direction = direction

    def __iter__(self):
        root, direction = self.graph._root, self.direction
        cur = getattr(root, direction)
        while cur is not root:
            yield cur
            cur = getattr(cur, direction)

class _Namespace:
    def __init__(self):
        self._used_names = set()

    def create_name(self, candidate):
        assert candidate not in self._used_names
        return candidate

class GraphModule(torch.nn.Module):
    def __init__(self, root, graph, class_name):
        super().__init__()
        self.__class__.__name__ = class_name
        self.graph = graph

        assert isinstance(root, torch.nn.Module)

        for node in graph.nodes:
            if node.node_type in ["call_module"]:
                assert isinstance(node.target, str)
                setattr(self, node.target, getattr(root, node.target))

    @property
    def code(self) -> str:
        return self._code

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, g):
        self._graph = g
        self.recompile()

    def gen_fn_def(self, fn_args):
        if len(fn_args) == 0 or fn_args[0] != "self":
            fn_args.insert(0, "self")
        return f"def forward({', '.join(fn_args)}):"

    def recompile(self):
        nodes = self.graph.nodes
        root_module = "self"
        body = []
        fn_args = [] # called free_vars in fx.graph.Codegen._gen_python_code

        def _format_target(base, target):
            return f"{base}.{target}"

        def _format_args(args, kwargs):
            args_s = ", ".join(repr(a) for a in args)
            kwargs_s = ", ".join(f"{k} = {repr(v)}" for k, v in kwargs.items())

            if args_s and kwargs_s:
                return f"{args_s}, {kwargs_s}"
            return args_s or kwargs_s

        def emit_node(node):
            if node.node_type == "placeholder":
                fn_args.append(node.target)
            elif node.node_type == "call_module":
                body.append(f"{repr(node)} = {_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})")
                # assert False, "call module"
            elif node.node_type == "output":
                assert len(node.args) == 1
                assert isinstance(node.args[0], Node)
                body.append(f"return {repr(node.args[0])}")
            else:
                raise RuntimeError(f"Could not handle node type: {node.node_type}")

        for node in nodes:
            emit_node(node)

        prologue = self.gen_fn_def(fn_args)
        code = "  " + "\n  ".join(body)
        fn_code = f"""
{prologue}
{code}"""
        self._code = fn_code

        globals_ = {}
        cls = type(self)

        exec(compile(fn_code, "", "exec"), globals_)
        cls.forward = globals_["forward"]

class Graph:
    def __init__(self):
        self._root = Node("", "root", "", (), {}) # this is a sentinel
        self._insert = self._root.prepend
        self._graph_namespace = _Namespace()

    def create_node(self, name, node_type, target, args, kwargs):
        assert name is None
        candidate = target
        name = self._graph_namespace.create_name(candidate)
        n = Node(name, node_type, target, args, kwargs)
        self._insert(n)
        return n

    @property
    def nodes(self):
        return _node_list(self)

    def __str__(self):
        placeholder_names = []
        node_strs = [nd.format_node(placeholder_names) for nd in self.nodes]
        param_str = ", ".join(placeholder_names)
        s = f"graph({param_str}):\n"
        for node_str in node_strs:
            if node_str:
                s += f"  {node_str}\n"
        return s

class Tracer:
    def __init__(self):
        self.graph = Graph()
        self.root = None

    def create_node(self, name, node_type, target, args, kwargs):
        return self.graph.create_node(name, node_type, target, args, kwargs)

    def unwrap_proxy(self, xargs):
        """
        The method is called create_arg in Fx
        """
        return tree_map(lambda x: x.node, xargs)
            
    def create_proxy(self, name, node_type, target, args, kwargs):
        args_ = self.unwrap_proxy(args)
        kwargs_ = self.unwrap_proxy(kwargs)
        node = self.create_node(name, node_type, target, args_, kwargs_)
        return Proxy(node)

    def path_of_module(self, mod):
        for p, m in self.root.named_modules():
            if m is mod:
                return p
        raise RuntimeError(f"Module not found: {mod}")

    def call_module(self, mod, forward, args, kwargs):
        """
        We don't need the forward method if we don't want to trace through
        """
        module_qualified_name = self.path_of_module(mod)
        return self.create_proxy(None, "call_module", module_qualified_name, args, kwargs)

    def trace(self, root):
        self.root = root
        fn = type(root).forward
        args = [root] # root is not a proxy
        code = fn.__code__
        assert code.co_kwonlyargcount == 0
        nargs = code.co_argcount + code.co_kwonlyargcount
    
        for i in range(1, nargs):
            args.append(self.create_proxy(None, "placeholder", code.co_varnames[i], (), {}))

        def module_call_wrapper(mod, *args, **kwargs):
            def forward(*args, **kwargs):
                return _orig_module_call(mod, *args, **kwargs)
            return self.call_module(mod, forward, args, kwargs)

        torch.nn.Module.__call__ = module_call_wrapper
        out_proxy = fn(*args)
        torch.nn.Module.__call__ = _orig_module_call
        self.create_node(None, "output", "", (self.unwrap_proxy(out_proxy),), {})
        return self.graph


def symbolic_trace(root: torch.nn.Module):
    tracer = Tracer()
    graph = tracer.trace(root)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return GraphModule(root, graph, name)
