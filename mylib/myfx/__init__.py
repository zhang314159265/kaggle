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
        raise RuntimeError("Module not found")

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
        self.create_node(None, "output", "", (self.unwrap_proxy(out_proxy),), {})
        return self.graph


def symbolic_trace(root: torch.nn.Module):
    tracer = Tracer()
    print(tracer.trace(root))
    import pdb; pdb.set_trace()
    # TODO: make sure the captured graph can be executed!
    assert False, "symbolic_trace ni"
