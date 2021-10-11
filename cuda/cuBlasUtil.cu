#include<cublas_v2.h>
#include"cuBlasUtil.h"
#include<stdio.h>

void gpu_blas_mmul(const float *A, const float *B, float *C, int m, int n, int k, bool transA, bool transB) {

        int lda=m,ldb=k,ldc=m;
        const float alf = 1;
        const float bet = 0;
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

