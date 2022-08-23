# from digit_recognizer.solution import ModelClass, get_example_batch
# from cifar10_object_recognization.solution import ModelClass, get_example_batch
from disaster_tweets.solution import ModelClass, get_example_batch
from torch import fx
from mylib import myfx
import torch

from mylib.torch_dispatch import torch_dispatch_trace

model = ModelClass()

# Example Fx trace for digit_recornizer DNN model
"""
graph(inp):
    %linear1 : [#users=1] = call_module[target=linear1](args = (%inp,), kwargs = {})
    %relu1 : [#users=1] = call_module[target=relu1](args = (%linear1,), kwargs = {})
    %linear2 : [#users=1] = call_module[target=linear2](args = (%relu1,), kwargs = {})
    return linear2

code generated for GraphModule:
def forward(self, inp):
    linear1 = self.linear1(inp);  inp = None
    relu1 = self.relu1(linear1);  linear1 = None
    linear2 = self.linear2(relu1);  relu1 = None
    return linear2
"""

cmd = "fx_trace"
# cmd = "myfx_trace"
# cmd = "dispatch_trace"
# cmd = "torch_package"

@torch.no_grad()
def try_trace(trace_method, model, example_inputs):
    print("Before tracing")
    if type(model).__module__.startswith("transformers."):
        # the transformer model
        concrete_args = {
            "inputs_embeds": None, 
            "attention_mask": None,
        }
        graph_module = trace_method(model, concrete_args=concrete_args)
    else:
        graph_module = trace_method(model)
    print("After tracing")
    print(str(graph_module.graph))
    print(f"code: {graph_module.code}")
    actual = graph_module(example_inputs)
    expected = model(example_inputs)
    assert torch.allclose(expected, actual)
    print(f"Pass {trace_method.__module__}.{trace_method.__name__}")

if cmd == "fx_trace":
    # Fx does not work for the transformer model
    try_trace(fx.symbolic_trace, model, get_example_batch(2))
elif cmd == "myfx_trace":
    try_trace(myfx.symbolic_trace, model, get_example_batch(2))
elif cmd == "dispatch_trace":
    print("Start dispatch_tree...")
    torch_dispatch_trace(model, get_example_batch(batch_size=2))
elif cmd == "torch_package":
    from torch.package import PackageExporter
    with PackageExporter("/tmp/my.torchpackage") as exporter:
        exporter.intern("digit_recognizer.solution")
        exporter.intern("mylib")
        for x in ["pandas", "numpy"]:
            exporter.extern(x)
        exporter.save_pickle("my_resources", "model.pkl", model)
    print("Try torch package done")
