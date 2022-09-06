LLVM_INSTALL="/home/shunting/.triton/llvm/clang+llvm-11.0.1-x86_64-linux-gnu-ubuntu-16.04"

LLVM_LIBS=-lLLVMCore -lLLVMRemarks -lLLVMBitstreamReader -lLLVMBinaryFormat -lLLVMSupport -lLLVMDemangle -lcurses -lLLVMIRReader -lLLVMMIRParser -lLLVMBitReader -lLLVMAnalysis -lLLVMAsmParser -lLLVMNVPTXCodeGen -lLLVMNVPTXDesc -lLLVMNVPTXInfo -lLLVMCodeGen -lLLVMMC -lLLVMTransformUtils -lLLVMTarget -lLLVMSelectionDAG -lLLVMAsmPrinter -lLLVMDebugInfoCodeView -lLLVMMCParser -lLLVMProfileData -lLLVMDebugInfoDWARF -lLLVMObject -lLLVMTextAPI -lLLVMScalarOpts -lLLVMVectorize -lLLVMipo -lLLVMFrontendOpenMP -lLLVMBitWriter -lLLVMInstrumentation -lLLVMInstCombine -lLLVMAggressiveInstCombine

all:
	g++ driver.cpp -lcuda -I$(LLVM_INSTALL)/include -L$(LLVM_INSTALL)/lib -Wl,--start-group $(LLVM_LIBS) -Wl,--end-group -lz
	./a.out

jit:
	g++ ptxjit.cpp -lcuda -lcudart
	./a.out

genptx:
	nvcc -ptx add.cu -allow-unsupported-compiler

as:
	# ptxas add.ptx -o add.cubin # this one can not be loaded by the driver. cuModuleLoadData fail
	ptxas --gpu-name=sm_86 add.ptx -o add.cubin
