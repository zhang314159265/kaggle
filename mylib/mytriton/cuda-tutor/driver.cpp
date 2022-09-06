#include <iostream>
#include <cuda.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <memory>
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IRReader/IRReader.h"

#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachine.h"

#define CHK(cufn) do { \
  auto code = cufn; \
  if (code != CUDA_SUCCESS) { \
    const char *msg; \
    cuGetErrorString(code, &msg); \
    std::cerr << "Fail to run '" #cufn "' (L" << __LINE__ << "), status code " << code << ", " << msg << std::endl; \
    exit(1); \
  } \
} while (0)

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction vecAdd_kernel;
CUdeviceptr d_A, d_B, d_C;

#if 0
std::string getLLIR() {
  std::ifstream ifs("triton_add_gen.ll");
  return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}
#endif

llvm::LLVMContext llvmContext;

std::unique_ptr<llvm::Module> parseLLIR() {
  llvm::SMDiagnostic Err;
  #if 0
  auto llModule = llvm::parseIRFile("triton_add_gen.ll", Err, llvmContext);
  #else
  auto llModule = llvm::parseIRFile("triton_add_manual.ll", Err, llvmContext);
  #endif
  // llModule->print(llvm::errs(), nullptr);
  return llModule;
}

void init_llvm() {
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
}

std::string llir_to_ptx(llvm::Module* module, int cc, int version) {
  { // setup options
    auto options = llvm::cl::getRegisteredOptions();
    auto* short_ptr = static_cast<llvm::cl::opt<bool>*>(options["nvptx-short-ptr"]);
    assert(short_ptr);
    short_ptr->setValue(true);
  }
  init_llvm();
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(*module);

  std::string triple = "nvptx64-nvidia-cuda";
  module->setTargetTriple(triple);

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);

  int max_nvvm_cc = 75;
  std::string proc = "sm_" + std::to_string(std::min(cc, max_nvvm_cc));
  std::string features;
  llvm::TargetOptions opt;
  {
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    opt.UnsafeFPMath = false;
    opt.NoInfsFPMath = false;
    opt.NoNaNsFPMath = true;
  }
  llvm::TargetMachine *machine = target->createTargetMachine(module->getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_, llvm::None, llvm::CodeGenOpt::Aggressive);
  module->setDataLayout(machine->createDataLayout());
  // emit machine code
  for (llvm::Function &f : module->functions()) {
    f.addFnAttr(llvm::Attribute::AlwaysInline);
  }
  llvm::legacy::PassManager pass;
  llvm::SmallVector<char, 0> buffer;
  llvm::raw_svector_ostream stream(buffer);
  // emit
  machine->addPassesToEmitFile(pass, stream, nullptr, llvm::CodeGenFileType::CGFT_AssemblyFile);
  pass.run(*module);

  // post process
  std::string result(buffer.begin(), buffer.end());
  // ignore a few string replacements for now
  std::cout << "Generated ptx:\n==========\n" << result << "\n==========\n" << std::endl;
  // assert(false && "llir_to_ptx ni"); // TODO
  return result;
}

void loadModuleFromLLIR() {
  // cc and version and dumped from triton
  int cc = 86;
  int version = 11060;

  std::unique_ptr<llvm::Module> llModule = parseLLIR();
  llvm::Module* module = llModule.get();

  std::string ptx_str = llir_to_ptx(module, cc, version);

  CHK(cuModuleLoadData(&cuModule, ptx_str.c_str()));
}

int main(int argc, char **argv) {
  int N = 50000;
  size_t size = N * sizeof(float);
  int devID = 0;
  CHK(cuInit(0));
  CHK(cuDeviceGet(&cuDevice, devID));
  CHK(cuCtxCreate(&cuContext, 0, cuDevice));

  // load from llir
  #if 1
  loadModuleFromLLIR();
  #endif

  // load ptx file
  #if 0
  // std::ifstream my_file("add.ptx");
  // std::ifstream my_file("add_manual.ptx");
  std::ifstream my_file("triton_add_gen.ptx");
  std::string my_ptx((std::istreambuf_iterator<char>(my_file)), std::istreambuf_iterator<char>());
  std::cout << "PTX: " << my_ptx << std::endl;
  // create module from ptx
  CHK(cuModuleLoadData(&cuModule, my_ptx.c_str()));
  #endif

  // load cubin
  #if 0
  std::ifstream my_file("add.cubin");
  std::string my_cubin((std::istreambuf_iterator<char>(my_file)), std::istreambuf_iterator<char>());
  CHK(cuModuleLoadData(&cuModule, my_cubin.c_str()));
  #endif

  // get function handle from module
  #if 1
  CHK(cuModuleGetFunction(&vecAdd_kernel, cuModule, "add"));
  #else
  CHK(cuModuleGetFunction(&vecAdd_kernel, cuModule, "add_kernel__Pfp32_Pfp32_Pfp32_i32__4c1024"));
  #endif

  // allocate/initialize vectors in host memory
  std::vector<float> h_A(N, 1.0f);
  std::vector<float> h_B(N, 2.0f);
  std::vector<float> h_C(N);

  // allocate vectors in device memory
  CHK(cuMemAlloc(&d_A, size));
  CHK(cuMemAlloc(&d_B, size));
  CHK(cuMemAlloc(&d_C, size));

  // copy vectors from host memory to device memory
  CHK(cuMemcpyHtoD(d_A, h_A.data(), size));
  CHK(cuMemcpyHtoD(d_B, h_B.data(), size));

  // grid/block configuration
  int threadsPerBlock = 128;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  void *args[] = { &d_A, &d_B, &d_C, &N };

  // launch the cuda kernel
  CHK(cuLaunchKernel(vecAdd_kernel, blocksPerGrid, 1, 1,
    threadsPerBlock, 1, 1,
    0,
    NULL, args, NULL));

  // copy result from device memory to host memory
  CHK(cuMemcpyDtoH(h_C.data(), d_C, size));

  // verify result
  bool mismatch_found = false;
  for (int i = 0; i < N; ++i) {
    float sum = h_A[i] + h_B[i];
    if (fabs(h_C[i] - sum) > 1e-7f) {
      std::cout << "mismatch " << i << ": " << h_C[i] << std::endl;
      mismatch_found = true;
      break;
    }
  }
  if (!mismatch_found) {
    std::cout << "Passed!" << std::endl;
  }
  std::cout << "bye" << std::endl;
  return 0;
}
