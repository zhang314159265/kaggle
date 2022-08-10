import torch
from contextlib import contextmanager

@contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard

class DispatchTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem: torch.Tensor):
        return torch.Tensor._make_subclass(cls, elem)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        with no_dispatch():
            res = func(*args, **kwargs)
        print(f"Call {func}")
        return res

@torch.no_grad()
def torch_dispatch_trace(model, inp):
    # TODO use pytree to handle container types
    assert isinstance(inp, torch.Tensor)
    inp_wrap = DispatchTensor(inp)
    return model(inp_wrap)

if __name__ == "__main__":
    x = DispatchTensor(torch.randn(5))
    y = DispatchTensor(torch.randn(5))
    z = x + y
    # print(z) # TODO: print will fail right now
