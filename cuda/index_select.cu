#pragma once
#include<cuda.h>

void index_select(float *src, int srcRows, int srcCols,
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
        dstTotalSize = indSize * srcNums;
    }
    else {
        printf("indexSelect kernel dimension error!\n");
        return;
    }
    
    // allocate device memory for output
    cudaMalloc(&out, dstTotalSize*sizeof(float));

    // dimensions of grids and blocks
    dim3 largeIndexGrid(dstTotalSize/128);
    dim3 largeIndexBlock(128);

    // launch kernel
    indexSelectLargeIndex<<<largeIndexGrid,largeIndexBlock>>>
                    (src,srcRows,srcCols,dim,indices,dstTotalSize
                    indSize, out);
    
}

__global__ indexSelectLargeIndex(float *src, int srcRows, int srcCols,
                    int dim, int* indices, int indSize, int dstSize
                    float *out) {
    
	const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx < dstSize) {

		// calculate the index info
		const id_row = *( indices + (int)(thread_idx/indSize) );
		const id_col = thread_idx % srcCols;

		// update respected cell
		*(out + threadid_x) = *(src + id_row*srcCols + id_col);
	}
    
}
