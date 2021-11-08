#include<iostream>
#include<cmath>
#include"C_GCN_MP.h"
#include<omp.h>

#define DIRECTED_EDGES 4

#define NUM_NODES 3

#define FEATURE_LEN 2


void GCNLayer(double edgeIndex[][4], double featureTensor[][2], double *aggregationVar, double *nodeDegrees) {

        for(int i=0; i<NUM_NODES; i++) {
                for(int j=0; j<DIRECTED_EDGES; j++) {
                        if(edgeIndex[0][j] == i) {// if there is an edge incoming to node i
                                // aggregate edgeIndex[1][j] features on node i
                                //std::cout << "from node " << edgeIndex[1][j] << " to node " << i << std::endl;
                                for(int k=0; k<FEATURE_LEN; k++) {
                                        aggregationVar[k] += featureTensor[i][k] * 1.0/sqrt(nodeDegrees[i]*nodeDegrees[(int)edgeIndex[1][j]]);
                                }
                        }
                }
                featureTensor[i][0] = aggregationVar[0];
                featureTensor[i][1] = aggregationVar[1];
                aggregationVar[0] = 0.0;
                aggregationVar[1] = 0.0;
        }

}

void GCNLayerNew(double* edgeIndex, double* featureTensor, double *aggregationVar, double *nodeDegrees) {

	#pragma omp parallel for num_threads(4)
        for(int i=0; i<NUM_NODES; i++) {
                for(int j=0; j<DIRECTED_EDGES; j++) {
                        if((*(edgeIndex + j)) == (double)i) {// if there is an edge incoming to node i
                                // aggregate edgeIndex[1][j] features on node i
                                //std::cout << "from node " << edgeIndex[1][j] << " to node " << i << std::endl;
                                for(int k=0; k<FEATURE_LEN; k++) {
                                        aggregationVar[k] += *(featureTensor + i*FEATURE_LEN + k) * 1.0/sqrt(nodeDegrees[i]*nodeDegrees[(int)(*(edgeIndex + 1*DIRECTED_EDGES + j))]);
                                }
                        }
                }
                *(featureTensor + FEATURE_LEN*i) = aggregationVar[0];
                *(featureTensor + FEATURE_LEN*i + 1) = aggregationVar[1];
                aggregationVar[0] = 0.0;
                aggregationVar[1] = 0.0;
        }


}
