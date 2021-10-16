#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


__global__ void GCNLayerNew(double* adjMatrix, double* featureTensor, int n_nodes, int n_edges, int n_features, double* output) {

    int i = threadIdx.x;
    if(i < n_nodes) {
        
        
        // ----- calculation of A^ ----- //
    	for(int i=0; i<n_nodes; i++) {
    		// add self loops
    		*(adjMatrix + (n_nodes+1)*i) += 1.0;
    	}
    
        
        // host side variable definitions
        
        int ld = n_nodes;
        int dense_size = ld * n_nodes;
        
        int   h_csr_offsets[] = { 0, 0, 0, 0, 0, 0  };
        int   h_csr_columns[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        float h_csr_values[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        double* h_dense;
        
        int   *d_csr_offsets, *d_csr_columns;
        double *d_csr_values,  *d_dense;
        
        // fill dense matrix with
        // input adjacent matrix values
        for(int i=0; i<(n_nodes*n_nodes); i++) {
            h_dense[i] = *(adjMatrix + i);
        }
        
        CHECK_CUDA( cudaMalloc((void**) &h_dense, dense_size * sizeof(double)))
        CHECK_CUDA( cudaMalloc((void**) &d_csr_offsets,
                           (num_nodes + 1) * sizeof(int)) )
        
        // cuSPARSE handle object
        cusparseHandle_t handle = NULL;
        
        // sparse matrix descriptor
        cusparseSpMatDescr_t matSparse;
        
        // dense matrix descriptor
        cusparseDnMatDescr_t matDense;
        
        // additional buffer
        void* dBuffer = NULL;
        
        // buffer size
        size_t bufferSize = 0;
        
        // create cuSPARSE handle
        CHECK_CUSPARSE( cusparseCreate(&handle) )
        
        // create dense matrix - input of conversion to CSR
        CHECK_CUSPARSE( cusparseCreateDnMat(&matDense, n_nodes, n_nodes, ld, h_dense,
                                            CUDA_R_32F, CUSPARSE_ORDER_ROW) )
        // create sparse matrix - output of conversion to CSR
        CHECK_CUSPARSE( cusparseCreateCsr(&matSparse, n_nodes, n_nodes, 0,
                                      d_csr_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
        
        // allocate an external buffer - optional
        CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        handle, matDense, matSparse,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) )
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
        
        // dense to sparse conversion
        CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, matDense, matSparse,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
        
         // get number of non-zero elements
        int64_t num_rows_tmp, num_cols_tmp, nnz;
        CHECK_CUSPARSE( cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,
                                         &nnz) )
        
        // allocate CSR column indices and values
        CHECK_CUDA( cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int))   )
        CHECK_CUDA( cudaMalloc((void**) &d_csr_values,  nnz * sizeof(float)) )
        // reset offsets, column indices, and values pointers
        CHECK_CUSPARSE( cusparseCsrSetPointers(matB, d_csr_offsets, d_csr_columns,
                                               d_csr_values) )
        
        
    	// calculation of D^-1/2
    	double* D = (double*)calloc(n_nodes*n_nodes, sizeof(double));
    	for(int i=0; i<n_nodes; i++) {
    		for(int j=0; j<=i; j++) {
    			*(D + (n_nodes+1)*i) += *(adjMatrix + i*n_nodes + j);
    		}
    		// square root
    		*(D + (n_nodes+1)*i) = sqrt((int)*(D + (n_nodes+1)*i));
    	}

    }
}
