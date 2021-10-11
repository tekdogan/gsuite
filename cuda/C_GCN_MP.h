void GCNLayer(float edgeIndex[][4], float featureTensor[][2], float *aggregationVar, float *nodeDegrees);
__global__ void GCNLayerNew(float* edgeIndex, float* featureTensor, float *aggregationVar, float *nodeDegrees);
