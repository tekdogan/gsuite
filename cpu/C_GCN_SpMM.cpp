#include<iostream>
#include<cmath>
#include<cstring>
#include"C_GCN_MP.h"
#include<omp.h>
#include"Data_Util.h"

#define DIRECTED_EDGES 4

#define NUM_NODES 3

#define FEATURE_LEN 2

namespace SpMM {

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

void GCNLayerNew(double* adjMatrix, double* featureTensor, int n_nodes, int n_edges, int n_features, double* output) {

	// calculation of D^-1/2
	double* D = (double*)calloc(n_nodes*n_nodes, sizeof(double));
	for(int i=0; i<n_nodes; i++) {
		for(int j=0; j<i; j++) {
			*(D + (n_nodes+1)*i) += *(adjMatrix + i*n_nodes + j);
		}
		// square root
		*(D + (n_nodes+1)*i) = sqrt((int)*(D + (n_nodes+1)*i));
	}


	// calculation of A^
	for(int i=0; i<n_nodes; i++) {
		// add self loops
		*(adjMatrix + (n_nodes+1)*i) += 1.0;
	}

	// D^-1/2 * A^
	double* DA = (double*)calloc(n_nodes*n_nodes, sizeof(double));
	matrix_mul(D, adjMatrix, DA, n_nodes, n_nodes, n_nodes, n_nodes);

	// D^-1/2 * A^ * D^-1/2
	double* DAD = (double*)calloc(n_nodes*n_nodes, sizeof(double));
	matrix_mul(DA, D, DAD, n_nodes, n_nodes, n_nodes, n_nodes);

	// D^-1/2 * A^ * D^-1/2 * X
	double* DADX = (double*)calloc(n_nodes*n_features, sizeof(double));
	matrix_mul(DAD, featureTensor, DADX, n_nodes, n_nodes, n_nodes, n_features);

	// copy the result to output
	memcpy(output, DADX, n_nodes*n_features);


}

}
