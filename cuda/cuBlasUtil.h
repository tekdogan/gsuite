void gpu_blas_mmul(const float *A, const float *B, float *C, int m, int k, int n, bool transA, bool transB, float alpha, float beta);

__global__ void initIdentityGPU(int **devMatrix, int numR, int numC);
