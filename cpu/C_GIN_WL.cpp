#include<iostream>
#include<cmath>
#include"C_GCN_MP.h"
#include<omp.h>

#define DIRECTED_EDGES 4

#define NUM_NODES 3

#define FEATURE_LEN 2





void GINLayer(double* edgeIndex, double* featureTensor, double *aggregationVar, double *nodeDegrees, double epsilon, double* featureTensorOutput) {

    #pragma omp parallel for num_threads(4)
    for(int i=0; i<NUM_NODES; i++) {
                for(int j=0; j<DIRECTED_EDGES; j++) {
                        if((*(edgeIndex + j)) == (double)i) {// if there is an edge incoming to node i
                                // aggregate edgeIndex[1][j] features on node i
                                //std::cout << "from node " << edgeIndex[1][j] << " to node " << i << std::endl;
                                for(int k=0; k<FEATURE_LEN; k++) {
                                        aggregationVar[k] += *(featureTensor + FEATURE_LEN*((int)(*(edgeIndex + 1*DIRECTED_EDGES + j))) + j);
                                }
                        }
                }
                *(featureTensorOutput + FEATURE_LEN*i) = aggregationVar[0] + (*(featureTensor + FEATURE_LEN*i))*(1+epsilon);
                *(featureTensorOutput + FEATURE_LEN*i + 1) = aggregationVar[1] + (*(featureTensor + FEATURE_LEN*i + 1))*(1+epsilon);
                aggregationVar[0] = 0.0;
                aggregationVar[1] = 0.0;
        }
}
