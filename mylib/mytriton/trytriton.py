import triton
import triton.language as tl
import torch

import triton._C.libtriton.triton as _triton

"""
triton ir for codegen:

def void add_kernel__Pfp32_Pfp32_Pfp32_i32__4c1024(f32* x_ptr .aligned(16), f32* y_ptr .aligned(16), f32* output_ptr .aligned(16), i32 n_elements .multipleof(16)){
entry:
  %0 = get_program_id(0) i32;
  %1 = mul i32 %0, 1024;
  %2 = make_range[0 : 1024] i32<1024>;
  %3 = splat i32<1024> %1;
  %4 = add i32<1024> %3, %2;
  %5 = splat i32<1024> n_elements;
  %6 = icmp_slt i1<1024> %4, %5;
  %7 = splat f32*<1024> x_ptr;
  %8 = getelementptr f32*<1024> %7, %4;
  %9 = splat f32<1024> undef;
  %10 = masked_load f32<1024> %8, %6, %9;
  %11 = splat f32*<1024> y_ptr;
  %12 = getelementptr f32*<1024> %11, %4;
  %13 = splat f32<1024> undef;
  %14 = masked_load f32<1024> %12, %6, %13;
  %15 = fadd f32<1024> %10, %14;
  %16 = splat f32*<1024> output_ptr;
  %17 = getelementptr f32*<1024> %16, %4;
  masked_store void %17, %15, %6;
  ret void;
}
"""
def tutor_add():
    print("Add kernel")
    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    def add(x: torch.Tensor, y: torch.Tensor):
        output = torch.empty_like(x)
        assert x.is_cuda and y.is_cuda and output.is_cuda
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        # kernel may still be running asynchronously after return
        return output

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f"max dif: {torch.max(torch.abs(output_torch - output_triton))}")

def tutor_softmax():
    print("softmax kernel")

    @torch.jit.script
    def naive_softmax(x):
        x_max = x.max(dim=1)[0]
        z = x - x_max[:, None]
        numerator = torch.exp(z)
        denominator = numerator.sum(dim=1)
        ret = numerator / denominator[:, None]
        return ret

    @triton.jit
    def softmax_kernel(
        output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

    def softmax(x):
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
        y = torch.empty_like(x)
        softmax_kernel[(n_rows,)](
            y,
            x,
            x.stride(0),
            y.stride(0),
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return y

    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")
    y_torch = torch.softmax(x, axis=1)
    y_naive = naive_softmax(x)
    y_triton = softmax(x)
    assert torch.allclose(y_torch, y_naive), (y_torch, y_naive)
    assert torch.allclose(y_torch, y_triton), (y_torch, y_triton)
    print("Pass tests!")

def tutor_matmul():
    print("matmul kernel: abit complex, skip for now")

# tutor_dummy()
tutor_add()
# tutor_softmax()
# tutor_matmul()
print("bye")
