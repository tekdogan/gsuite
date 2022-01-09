#include"cusparse_v2.h"

namespace CU_MP {

void GCNLayer(float* edgeIndex, float* featureTensor, float *aggregationVar, float *nodeDegrees,
		int numOfNodes, int numOfFeatures, int numOfEdges, int outputSize);
__global__ void GCNLayerNew(float* edgeIndex, float* featureTensor, float *aggregationVar, float *nodeDegrees, int numOfNodes, int numOfFeatures, int numOfEdges);

}
