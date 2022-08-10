from digit_recognizer.solution import ModelClass, get_example_batch
from torch import fx
import torch

from mylib.torch_dispatch import torch_dispatch_trace

def fx_trace(model):
    graph_module = fx.symbolic_trace(model)
    print(str(graph_module.graph))

model = ModelClass()
# fx_trace(model)
# Got:
"""
graph(inp):
    %linear1 : [#users=1] = call_module[target=linear1](args = (%inp,), kwargs = {})
    %relu1 : [#users=1] = call_module[target=relu1](args = (%linear1,), kwargs = {})
    %linear2 : [#users=1] = call_module[target=linear2](args = (%relu1,), kwargs = {})
    return linear2
"""

torch_dispatch_trace(model, get_example_batch(batch_size=2))
