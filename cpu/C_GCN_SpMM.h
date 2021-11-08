namespace SpMM {

void GCNLayer(double edgeIndex[][4], double featureTensor[][2], double *aggregationVar, double *nodeDegrees);
void GCNLayerNew(double* adjMatrix, double* featureTensor, int n_nodes, int n_edges, int n_features, double* output);

}
