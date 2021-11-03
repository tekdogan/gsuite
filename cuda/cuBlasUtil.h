void gpu_blas_mmul(const float *A, const float *B, float *C, int m, int k, int n, bool transA, bool transB, float alpha, float beta);

void initIdentityMatrix(float* matrix, int R, int C);

__global__ void initIdentityGPU(float **devMatrix, int numR, int numC);
