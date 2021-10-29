#include<cublas_v2.h>
#include"cuBlasUtil.h"
#include<stdio.h>

void gpu_blas_mmul(const float *A, const float *B, float *C, int m, int n, int k, bool transA, bool transB, float alpha, float beta) {

        int lda=m,ldb=k,ldc=m;
        const float alf = alpha;
        const float bet = beta;
        const float *alpha = &alf;
        const float *beta = &bet;

        cublasHandle_t handle;
        cublasCreate(&handle);

	cublasOperation_t tA, tB;
	if(transA) {
		tA = CUBLAS_OP_T;
	}
	else {
		tA = CUBLAS_OP_N;
	}
	if(transB) {
                tB = CUBLAS_OP_T;
        }
        else {
                tB = CUBLAS_OP_N;
        }

	cublasStatus_t status = cublasSgemm(handle, tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS operation failed\n");
	}
	else {
		printf("CUBLAS operation is successful!\n");
	}

        cublasDestroy(handle);
}


__global__ void initIdentityGPU(int **devMatrix, int numR, int numC) {
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if(y < numR && x < numC) {
          if(x == y)
              devMatrix[y][x] = 1;
          else
              devMatrix[y][x] = 0;
    }
}
