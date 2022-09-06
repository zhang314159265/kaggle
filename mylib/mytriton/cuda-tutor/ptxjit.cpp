#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#define CHK(cufn) do { \
  auto code = cufn; \
  if (code != CUDA_SUCCESS) { \
    const char *msg; \
    cuGetErrorString(code, &msg); \
    std::cerr << "Fail to run '" #cufn "' (L" << __LINE__ << "), status code " << code << ", " << msg << std::endl; \
    exit(1); \
  } \
} while (0)

// normally this should be loaded from a ptx file
std::string ptx_source = R"PTX(
.version 7.6
.target sm_52
.address_size 64

	// .globl	myKernel

.visible .entry myKernel(
	.param .u64 myKernel_param_0
)
{
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [myKernel_param_0];
	cvta.to.global.u64 	%rd2, %rd1;
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	mul.wide.s32 	%rd3, %r4, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.u32 	[%rd4], %r4;
	ret;

}

)PTX";

void ptxJIT(int argc, char **argv, CUmodule *phModule, CUfunction *phKernel, CUlinkState *lState) {
	#define NOPT 6
	#define LOG_SIZE 8192
	CUjit_option options[NOPT];
	void *optionVals[NOPT];
	float walltime;
	char info_log[LOG_SIZE], error_log[LOG_SIZE];
	void *cuOut;
	size_t outSize;

	// Setup linker options
	// return walltime from JIT compilation
	options[0] = CU_JIT_WALL_TIME;
	optionVals[0] = (void *)&walltime;
	// pass a buffer for info messages
	options[1] = CU_JIT_INFO_LOG_BUFFER;
	optionVals[1] = (void *)info_log;
	// pass the size of the info buffer
	options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
	optionVals[2] = (void *)(long) LOG_SIZE;
	// pass a buffer for error message
	options[3] = CU_JIT_ERROR_LOG_BUFFER;
	optionVals[3] =(void *) error_log;
	// pass the size of the error buffer
	options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
	optionVals[4] = (void *)(long) LOG_SIZE;
	// make the linker verbose
	options[5] = CU_JIT_LOG_VERBOSE;
	optionVals[5] = (void *) 1;

	// create a pending linker invocation
	CHK(cuLinkCreate(NOPT, options, optionVals, lState));

	// TODO: print error_log if fail
	CHK(cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void *) ptx_source.c_str(),
		ptx_source.size() + 1, 0, 0, 0, 0));

	// complete the linker step
	CHK(cuLinkComplete(*lState, &cuOut, &outSize));

	std::cout << "CUDA Link Completed in " << walltime << "ms. Linker output:\n"
		<< info_log << std::endl;

	// Load resulting cuBin info module
	CHK(cuModuleLoadData(phModule, cuOut));

	// locate the kernel entry point
	CHK(cuModuleGetFunction(phKernel, *phModule, "myKernel"));

	// destroy the linker invocation
	CHK(cuLinkDestroy(*lState));
}

void setupDevice() {
	int device_count = 0;
	int devID = 0;
	char name[100];
	CUdevice cuDevice;
	CHK(cuInit(0));
	CHK(cuDeviceGetCount(&device_count));
	std::cout << "Found " << device_count << " devices" << std::endl;
	assert(device_count > 0);

	CHK(cuDeviceGet(&cuDevice, devID));
	CHK(cuDeviceGetName(name, sizeof(name), cuDevice));
	std::cout << "> Using CUDA device [" << devID << "]: " << name << std::endl;

	CUcontext cuContext;
	CHK(cuCtxCreate(&cuContext, 0, cuDevice));
}

int main(int argc, char **argv) {
  CUmodule hModule = 0;
  CUfunction hKernel = 0;
  CUlinkState lState;
	const unsigned int nThreads = 256;
	const unsigned int nBlocks = 64;
	const size_t memSize = nThreads * nBlocks * sizeof(int);
	int *h_data = 0;
	CUdeviceptr d_data;

	setupDevice(); // a critical step
	int driverVersion;
	cudaDriverGetVersion(&driverVersion);
  std::cout << "driverVersion = " << driverVersion << " v.s. "
	  << "CUDART_VERSION = " << CUDART_VERSION << std::endl;
	if (driverVersion < CUDART_VERSION) {
		exit(1);
	}
  ptxJIT(argc, argv, &hModule, &hKernel, &lState);

	dim3 block(nThreads, 1, 1);
	dim3 grid(nBlocks, 1, 1);

	h_data = (int*) malloc(memSize);
	// CHK(cudaMalloc(&d_data, memSize));
	CHK(cuMemAlloc(&d_data, memSize));

	void *args[1] = {&d_data};
	CHK(cuLaunchKernel(hKernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL));
	std::cout << "CUDA kernel launched" << std::endl;

	// CHK(cudaMemcpy(h_data, d_data, memSize, cudaMemcpyDeviceToHost));
	CHK(cuMemcpyDtoH(h_data, d_data, memSize));
	for (int i = 0; i < nBlocks * nThreads; ++i) {
		if (h_data[i] != i) {
			std::cout << "Error at " << i << ", bad value " << h_data[i] << std::endl;
			exit(1);
		}
	}
	std::cout << "PASS!" << std::endl;

	// CHK(cudaFree(d_data));
	CHK(cuMemFree(d_data));
	free(h_data);
	CHK(cuModuleUnload(hModule));

  std::cout << "bye" << std::endl;
  return 0;
}
