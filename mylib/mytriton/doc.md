# Note
It seems Triton does not work on Py3.9 (either debug or release build). The 'add' tutorial program crashes in the CPython interpreter complaining memory allocation/deallocation happens without holding the GIL.

I can try triton out on Py3.7 though.

# Achievement
Understand the triton lowering pipeline in high level. A python AST is converted to triton ir, then LLVM IR and then PTX. PTX string is converted to CUDA runnable. I'm able to dump all these 4 representations: python AST (ast.dump), triton ir (ir.print), LLVM IR and PTX string. 

# PTX
Command `nvcc -ptx input.cu` will generate input.ptx. Note that host code are not included here.

ptxas can assemble a PTX file. Example command
`ptxas -v --gpu-name=sm_86 input.ptx -o output.o`

# MISC
- cubin v.s. fatbin: it looks like cubin contains device binary code for one device while fatbin contains device binary code for multiple devices.
- driver.cpp can be used to load and run kernel from both ptx and cubin file.

# References
- [triton openai blog](https://openai.com/blog/triton/): Describes some of the architecture of triton.
- [triton official documentation](https://triton-lang.org/master/index.html): contains some awesome tutorials.
- [Cuda is a Low-Level Language - youtube](https://www.youtube.com/watch?v=KHa-OSrZPGo): awesome talk.
- [A very good reference to write CPU driver code to load ptx/cubin](https://stackoverflow.com/questions/67752857/how-can-i-create-an-executable-to-run-a-kernel-in-a-given-ptx-file): the reference pointers to NVIDIA official cuda samples. The ones mentioend (driver & jit) are all super good.
- [PTX ISA - nvidia](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html): use this as a lookup table
- [llvm NVPTX backend](https://llvm.org/docs/NVPTXUsage.html)

# TODO
- document the behavior of `__syncthreads` (sync all threads in block?)

# Scratch

- TODO: understand how triton IR is converted to the LLIR ... <===
  - emit llir comment to show boundary for each ttir instruction (or create some sort of annotatoin) so I know which ttir instruction generate each of the llir instruction
  - run trytriton.py
 
  - NOTE: driver.cpp is able to run with a manually created llvm IR.

- NVVM IR Spec: https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html
  - found by searching 'nvvm.annotations' in google
  - stop at 3.2 TODO HERE

- [triton paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
