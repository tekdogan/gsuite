
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cuda.h>
#include <cublas_v2.h>
#include "CU_SpMM_GIN.h"
#include "cuBlasUtil.h"
#include "Data_Util.h"


namespace CU_SpMM {


void GINLayer(float* adjMatrix, float* featureTensor, int n_nodes, int n_edges, int n_features, float* output, float epsilon) {

//    int i = threadIdx.x;
//    if(i < n_nodes) {
        

	// create identity matrix I here
	float *d_I, *d_A, *d_AIX, *d_X;
	cudaMalloc(&d_I,n_nodes * n_nodes * sizeof(float));
	initIdentityGPU<<<16,1024>>>(&d_I, n_nodes, n_nodes);
        
        // ----- calculation of (1+e)*I ----- //
	//for(int i=0; i<n_nodes; i++) {

	//}

	printf("I matrix:\n");
	float* I = (float*)calloc(n_nodes * n_nodes, sizeof(float));
	//cudaMemcpy(I,d_I,n_nodes * n_nodes * sizeof(float), cudaMemcpyDeviceToHost);
	initIdentityMatrix(I, n_nodes, n_nodes);
	printDenseMatrix(I, n_nodes, n_nodes);

	printf("A matrix:\n");
	printDenseMatrix(adjMatrix, n_nodes, n_nodes);

	// allocate device A matrix
        cudaMalloc(&d_A,n_nodes * n_nodes * sizeof(float));
                
	// migrate A matrix to device
	cudaMemcpy(d_A,adjMatrix,n_nodes * n_nodes * sizeof(float),cudaMemcpyHostToDevice);

	// ----- calculation of A + (1+e)*I ----- //
	gpu_blas_mmul(d_I, d_A, d_I, n_nodes, n_nodes, n_nodes, false, false, 1.0, (1.0 + epsilon));
	cudaFree(d_A);

	// migrate node feature values from host to device
	cudaMalloc(&d_X, n_nodes * n_features * sizeof(float));
	cudaMemcpy(d_X, featureTensor, n_nodes * n_features * sizeof(float), cudaMemcpyHostToDevice);

	// ----- calculation of (A + (1+e)*I) * X ----- //
	cudaMalloc(&d_AIX,n_nodes * n_features * sizeof(float));
	gpu_blas_mmul(d_X, d_I, d_AIX, n_features, n_nodes, n_features, false, false, 1.0, 0.0);
	cudaFree(d_I);
	cudaFree(d_X);

	// copy the result to output
	cudaMemcpy(output, d_AIX, n_nodes * n_features * sizeof(float), cudaMemcpyDeviceToHost);
	memcpy(output, d_AIX, sizeof(float)*(n_nodes*n_features));

	//printf("DADX matrix:\n");
	//printDenseMatrix(DADX, n_nodes, n_features);

	cudaFree(d_AIX);
//    }

}

} //namespace end
