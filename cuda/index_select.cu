#include "index_select.h"

float* index_select(float *src, int srcRows, int srcCols,
                    int dim, int* indices, int indSize,
                    float *out) {

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
    cudaMalloc(&out, dstTotalSize*sizeof(float));

    // dimensions of grids and blocks
    dim3 largeIndexGrid(dstTotalSize/128);
    dim3 largeIndexBlock(128);

    // launch kernel
    indexSelectLargeIndex<<<largeIndexGrid,largeIndexBlock>>>
                    (src,srcRows,srcCols,dim,indices,indSize,
			dstTotalSize,out);

    return out;
}

__global__ void indexSelectLargeIndex(float *src, int srcRows, int srcCols,
                    int dim, int* indices, int indSize, int dstSize,
                    float *out) {
    
	const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx < dstSize) {

		// calculate the index info
		const int id_row = (int)*( indices + (int)(thread_idx/indSize) );
		const int id_col = thread_idx % srcCols;

		// update respected cell
		*(out + thread_idx) = *(src + id_row*srcCols + id_col);
	}
    
}
