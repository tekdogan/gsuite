#include<cublas_v2.h>
#include"cuBlasUtil.h"
#include<stdio.h>

void gpu_blas_mmul(const float *A, const float *B, float *C, int m, int n, int k, bool transA, bool transB, float Alpha, float Beta) {

        int lda=m,ldb=k,ldc=m;
        const float alf = Alpha;
        const float bet = Beta;
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

	// plain sgemm
	//cublasStatus_t status = cublasSgemm(handle, tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	
	// sgemm with precisions indicated
	cublasStatus_t status = cublasSgemmEx(handle, tA, tB, m, n, k, alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, beta, C, CUDA_R_32F, ldc);
	
	// batched sgemm
	//cublasStatus_t status = cublasSgemmBatched(handle, tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 32);

	cudaDeviceSynchronize();

	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS operation failed\n");
	}
	else {
		printf("CUBLAS operation is successful!\n");
	}

        cublasDestroy(handle);
}


__global__ void initIdentityGPU(float **devMatrix, int numR, int numC) {
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if(y < numR && x < numC) {
          if(x == y)
              devMatrix[y][x] = 1;
          else
              devMatrix[y][x] = 0;
    }
}


void initIdentityMatrix(float* matrix, int R, int C) {

	for(int i=0; i<R; i++) {
		for(int j=0; j<C; j++) {
			if(i == j) {
				*( matrix + i*C + j ) = 1.0;
			}
			else {
				*( matrix + i*C + j ) = 0.0;
			}
		}
	}

}
