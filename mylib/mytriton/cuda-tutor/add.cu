#include <iostream>

extern "C" __global__ void add(const float* a, const float *b, float *c, int N) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N) {
	  c[idx] = a[idx] + b[idx];
  }
}

#if 1
int main(void) {
  float a[10], b[10], c[10];
  // init the cpu buffer
  for (int i = 0; i < 10; ++i) {
    a[i] = i;
    b[i] = i * 10;
  }
  // copy cpu buffer to gpu buffer
  float *agpu, *bgpu, *cgpu;
  cudaMalloc((void **) &agpu, sizeof(a));
  cudaMalloc((void **) &bgpu, sizeof(b));
  cudaMalloc((void **) &cgpu, sizeof(c));
  cudaMemcpy(agpu, a, sizeof(a), cudaMemcpyHostToDevice);
  cudaMemcpy(bgpu, b, sizeof(b), cudaMemcpyHostToDevice);

  add<<<10, 1>>>(agpu, bgpu, cgpu, 10);
  cudaMemcpy(c, cgpu, sizeof(c), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; ++i) {
    std::cout << c[i] << std::endl;
  }
  return 0;
}
#endif
