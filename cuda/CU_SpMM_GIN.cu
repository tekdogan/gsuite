
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cuda.h>
#include <cublas_v2.h>
#include "CU_SpMM_GCN.h"
#include "cuBlasUtil.h"
#include "Data_Util.h"


namespace CU_SpMM {


void GCNLayer(float* adjMatrix, float* featureTensor, int n_nodes, int n_edges, int n_features, float* output) {

//    int i = threadIdx.x;
//    if(i < n_nodes) {
        

	// create identity matrix I here
	float *d_I;
	cudaMalloc(&d_I,n_nodes * n_nodes * sizeof(float));
	initIdentityGPU<<<16,1024>>>(d_I, n_nodes, n_nodes);
        
        // ----- calculation of (1+e)*I ----- //
	//for(int i=0; i<n_nodes; i++) {

	//}

	//printf("A matrix:\n");
	//printDenseMatrix(adjMatrix, n_nodes, n_nodes);

	// define device matrices
        float *d_A, *d_D, *d_DA, *d_DAD, *d_DADX, *d_X;

	// allocate device A matrix
        cudaMalloc(&d_A,n_nodes * n_nodes * sizeof(float));
                
	// migrate A matrix to device
	cudaMemcpy(d_A,adjMatrix,n_nodes * n_nodes * sizeof(float),cudaMemcpyHostToDevice);

	// ----- calculation of A + (1+e)*I ----- //
	float* D = (float*)calloc(n_nodes*n_nodes, sizeof(float));
	gpu_blas_mmul(d_I, d_A, d_AI, n_nodes, n_nodes, n_nodes, false, false);


	// ----- calculation of (A + (1+e)*I) * X ----- //
	cudaMalloc(&d_DA,n_nodes * n_nodes * sizeof(float));
	gpu_blas_mmul(d_A, d_D, d_DA, n_nodes, n_nodes, n_nodes, false, false);
	float* DA = (float*)calloc(n_nodes*n_nodes, sizeof(float));
	//gpu_blas_mmul(D, adjMatrix, DA, n_nodes, n_nodes, n_nodes);
	cudaFree(d_A);

	cudaMemcpy(DA,d_DA,n_nodes * n_nodes * sizeof(float),cudaMemcpyDeviceToHost);
	//printf("DA matrix:\n");
	//printDenseMatrix(DA, n_nodes, n_nodes);
        
	// ----- calculation of D^-1/2 * A^ * D^-1/2 ----- //
	cudaMalloc(&d_DAD,n_nodes * n_nodes * sizeof(float));
	gpu_blas_mmul(d_D, d_DA, d_DAD, n_nodes, n_nodes, n_nodes, false, false);
        cudaFree(d_DA);
        cudaFree(d_D);

	float* DAD = (float*)calloc(n_nodes*n_nodes, sizeof(float));
	cudaMemcpy(DAD,d_DAD,n_nodes * n_nodes * sizeof(float),cudaMemcpyDeviceToHost);
	//printf("DAD matrix:\n");
	//printDenseMatrix(DAD, n_nodes, n_nodes);
        
	// migrate node feature values from host to device
	cudaMalloc(&d_X, n_nodes * n_features * sizeof(float));
	cudaMemcpy(d_X, featureTensor, n_nodes * n_features * sizeof(float), cudaMemcpyHostToDevice);

	//printf("X matrix:\n");
	//printDenseMatrix(featureTensor, n_nodes, n_features);

	// ----- calculation of D^-1/2 * A^ * D^-1/2 * X ----- //
	cudaMalloc(&d_DADX,n_nodes * n_features * sizeof(float));
	gpu_blas_mmul(d_X, d_DAD, d_DADX, n_features, n_nodes, n_features, false, false);
	cudaFree(d_DAD);
	cudaFree(d_X);

	// copy the result to output
	float* DADX = (float*)calloc(n_nodes * n_features, sizeof(float));
	cudaMemcpy(DADX, d_DADX, n_nodes * n_features * sizeof(float), cudaMemcpyDeviceToHost);
	memcpy(output, DADX, sizeof(float)*(n_nodes*n_features));

	//printf("DADX matrix:\n");
	//printDenseMatrix(DADX, n_nodes, n_features);
//    }

}

} //namespace end
