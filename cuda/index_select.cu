


void index_select(float *src, int srcRows, int srcCols,
                    int dim, int* indices, int indSize,
                    float *out) {
    int dstTotalSize;
    
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
    
    dim3 largeIndexGrid(dstTotalSize/128);
    dim3 largeIndexBlock(128);
                        
    indexSelectLargeIndex<<<largeIndexGrid,largeIndexBlock>>>
                    (src,srcRows,srcCols,dim,indices,dstTotalSize
                    indSize, out);
    
}

__global__ indexSelectLargeIndex(float *src, int srcRows, int srcCols,
                    int dim, int* indices, int indSize, int dstSize
                    float *out) {
    
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: calculate the index info below
    const int indexInfo = 1;

    if (thread_idx < dstSize) {

        *(out + threadid_x) = *(src + indexInfo);
    
    }
    
}
