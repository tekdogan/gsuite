#include "index_select.h"

float* index_select(float *src, int srcRows, int srcCols,
                    int dim, int* indices, int indSize) {

	float *d_src, *d_out;
	int *d_indices;

	float *out;
	out = (float*)calloc(indSize*srcCols, sizeof(float));

	cudaMalloc((void**) &d_src, srcRows*srcCols*sizeof(float));
	cudaMemcpy(d_src, src, srcRows*srcCols*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_indices, indSize*sizeof(int));
	cudaMemcpy(d_indices, indices, indSize*sizeof(int), cudaMemcpyHostToDevice);
	


    int dstTotalSize;
    
    // calculate output size
    // index along x-axis
    if(dim == 0) {
        dstTotalSize = indSize * srcCols;
    }
    // index along y-axis
    else if(dim == 1){
        dstTotalSize = indSize * srcRows;
    }
    else {
        printf("indexSelect kernel dimension error!\n");
        return src;
    }
    
    // allocate device memory for output
    cudaMalloc((void**) &d_out, dstTotalSize*sizeof(float));

    // dimensions of grids and blocks
    dim3 largeIndexGrid(dstTotalSize);
    dim3 largeIndexBlock(1);

    // launch kernel
    indexSelectLargeIndex<<<largeIndexGrid,largeIndexBlock>>>
                    (d_src,srcRows,srcCols,dim,d_indices,indSize,
			dstTotalSize,d_out);

    cudaMemcpy(out, d_out, dstTotalSize*sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(d_src);
    cudaFree(d_indices);
    cudaFree(d_out);
    return out;
}

__global__ void indexSelectLargeIndex(float *src, int srcRows, int srcCols,
                    int dim, int* indices, int indSize, int dstSize,
                    float *out) {
    
	const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx < dstSize) {

		// calculate the index info
		const int id_row = (int)*( indices + (int)(thread_idx/srcCols) );
		const int id_col = thread_idx % srcCols;

//		printf("thread id: %d, id_row: %d, id_col, %d\n", thread_idx, id_row, id_col);
		// update respected cell

		int data = *(src + id_row*srcCols + id_col);

		*(out + thread_idx) = data;
	}
    
}
