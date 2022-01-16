#include"cusparse_v2.h"

namespace CU_MP {

float* SAGELayer(int* h_edgeIndex, float* h_featureVector, int numOfNodes, int numOfFeatures,
                        int numOfEdges, int outputSize);
}
