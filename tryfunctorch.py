import torch
from functorch import vmap, grad, make_fx
from functorch.compile import aot_function, make_boxed_compiler
from functorch.experimental import functionalize

def first_tutor():
    x = torch.randn(3)
    y = vmap(torch.sin)(x)
    assert torch.allclose(y, x.sin())

def second_tutor():
    batch_size, feature_size = 3, 5
    weights = torch.randn(feature_size, requires_grad=True)
    
    def model(feature_vec):
        # Very simple linear model with activation
        assert feature_vec.dim() == 1
        return feature_vec.dot(weights).relu()
    
    examples = torch.randn(batch_size, feature_size)
    result = vmap(model)(examples)
    print(f"Result is {result}")

def third_tutor():
    x = torch.randn([])
    cos_x = grad(lambda x: torch.sin(x))(x)
    assert torch.allclose(cos_x, x.cos())
    
    # Second-order gradients
    neg_sin_x = grad(grad(lambda x: torch.sin(x)))(x)
    assert torch.allclose(neg_sin_x, -x.sin())
    print("Pass third tutor")

def fourth_tutor():
    """aot autograd"""
    fn = lambda x : x.sin().cos()
    @make_boxed_compiler
    def print_compile_fn(fx_module, args):
        print(fx_module)
        return fx_module
    aot_fn = aot_function(fn, print_compile_fn)
    x = torch.randn(4, 5, requires_grad=True)
    aot_fn(x).sum().backward()

def fifth_tutor():
    """functionalization"""
    def f(a):
        b = a + 1
        c = b.view(-1)
        c.add_(1)
        return b

    inpt = torch.randn(2)

    out1 = f(inpt)
    out2 = functionalize(f)(inpt);
    print(torch.allclose(out1, out2))

    f_traced = make_fx(f)(inpt)
    f_no_mutations_traced = make_fx(functionalize(f))(inpt)
    f_no_mutations_and_views_traced = make_fx(functionalize(f, remove="mutations_and_views"))(inpt)

    print(f_traced.code)
    print(f_no_mutations_traced.code)
    print(f_no_mutations_and_views_traced.code)

fifth_tutor()
