namespace CU_WL {

float* GINLayer(float* edgeIndex, float* featureTensor, float *aggregationVar, float epsilon, int numOfNodes,
	      int numOfDirectedEdges, int numOfFeatures, float* outputFeatureMatrix, int outputSize);

}
