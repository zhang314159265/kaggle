import torch
from torch import nn
import torchdynamo
import mydynamo
from typing import List
from digit_recognizer.solution import ClassifierDNN as ModelClass, get_example_batch

def orig_fn(x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    # why calling misc.sub cause sin/cos disappearing from the graph?
    # res = misc.sub(a, b)
    res = a - b
    return res

def run(dynamo_mod, orig_fn, inputs, backend=None):
    print(f"Try {dynamo_mod}")

    def compiler_fn(gm, example_inputs):
        print("Compile graph:")
        gm.graph.print_tabular()
        print(f"  with inputs: {example_inputs}")
        print(f"Code:\n{gm.code}")
        return gm.forward

    if backend is None:
        backend = compiler_fn

    opt_fn = dynamo_mod.optimize(backend)(orig_fn)
    actual_res = opt_fn(*inputs)
    expected_res = orig_fn(*inputs)
    assert torch.allclose(expected_res, actual_res)
    actual_res.sum().backward() # trigger the backward graph compilation
    print(f"Pass test! Result is {actual_res}")

# fn_args = [orig_fn, (torch.randn(10), torch.randn(10))]
fn_args = [ModelClass(), [get_example_batch(2),]]

from torchinductor import config
config.debug = True
run([torchdynamo, mydynamo][
0
], *fn_args, backend="inductor")
