#include"DataLoader.h"

#define SIZE 1024

#ifdef __cplusplus
extern "C" {
#endif

int LoadData() {

	float *h_edgeIndex, *h_featureVector;
	float *edgeIndex, *featureVector;

	float h_aggregationVar[2] = {0,0};
	float h_nodeDegrees[3] = {1,2,1};

	float aggregationVar[2], nodeDegrees[3];

	std::unordered_map<int,int> nodeMap;

	cudaMallocManaged( (void**) &aggregationVar, 2 * sizeof(float));
	cudaMemcpy(aggregationVar, h_aggregationVar, 2*sizeof(float), cudaMemcpyHostToDevice);

	cudaMallocManaged( (void**) &nodeDegrees, 3*sizeof(float));
	cudaMemcpy(nodeDegrees, h_nodeDegrees, 3*sizeof(float), cudaMemcpyHostToDevice);

	const char* edgeIndexFileName = "cora.cites.bak2";
	int edgeIndexSize = getEdgeIndexSizeFromFile(edgeIndexFileName);
	std::cout << "edgeIndexSize: " << edgeIndexSize << std::endl;

	const char* featureFileName = "cora.content.bak2";
	int featureSize = getFeatureSizeFromFile(featureFileName);
	std::cout << "featureSize: " << featureSize << std::endl;

	int numOfNodes = getNumOfNodesFromFile(featureFileName);
	std::cout << "numOfNodes: " << numOfNodes << std::endl;

	try {
		h_featureVector = (float*) calloc(numOfNodes*featureSize, sizeof(float));
		loadFeatureVectorFromFile(featureFileName, h_featureVector, featureSize, nodeMap);
		cudaMallocManaged( (void**) &featureVector, numOfNodes*featureSize * sizeof(float));
		cudaMemcpy(featureVector, h_featureVector, numOfNodes*featureSize * sizeof(float), cudaMemcpyHostToDevice);
	} catch(...) {
		std::cout << "Could not allocate memory space for featureVector!\n";
	}
	
	try {
		h_edgeIndex = (float*) calloc(featureSize*edgeIndexSize, sizeof(float));
		loadEdgeIndexFromFile(edgeIndexFileName, h_edgeIndex, edgeIndexSize, nodeMap);
		cudaMallocManaged( (void**) &edgeIndex, featureSize*edgeIndexSize * sizeof(float));
		cudaMemcpy(edgeIndex, h_edgeIndex, (size_t)featureSize*edgeIndexSize*sizeof(float), cudaMemcpyHostToDevice);
		
	} catch(...) {
		std::cout << "Could not allocate memory space for edgeIndex!\n";
	}

	auto start = std::chrono::steady_clock::now();

	//CU_MP::GCNLayerNew<<<16,SIZE>>>(edgeIndex, featureVector, aggregationVar, nodeDegrees);

	//CU_MP::GCNLayerNew<<<16,SIZE>>>(edgeIndex, featureVector, aggregationVar, nodeDegrees);

	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double, std::milli> dur_ms = end-start;
	std::cout << "2-layer GCN execution took " << dur_ms.count() << " ms\n";

	//for(int i=0; i<3; i++) {
	//	std::cout << "Node" << i << " feature 0: " << *(featureVector + featureSize*i) << std::endl;
	//	std::cout << "Node" << i << " feature 1: " << *(featureVector + featureSize*i + 1) << std::endl;
	//}


	float *featureVectorOutput;
	cudaMallocManaged( (void**) &featureVectorOutput, numOfNodes*featureSize * sizeof(float));
        start = std::chrono::steady_clock::now();
        //GINLayer<<<1,SIZE>>>(edgeIndex, featureVector, aggregationVar, nodeDegrees, 1.0, featureVectorOutput);
        end = std::chrono::steady_clock::now();
        dur_ms = end-start;
        std::cout << "1-layer GIN execution took " << dur_ms.count() << " ms\n";

	//for(int i=0; i<3; i++) {
        //        std::cout << "Node" << i << " feature 0: " << *(featureVectorOutput + featureSize*i) << std::endl;
        //        std::cout << "Node" << i << " feature 1: " << *(featureVectorOutput + featureSize*i + 1) << std::endl;
        //}

	//float *adjMatrix = (float*)calloc(numOfNodes*numOfNodes, sizeof(float));
	float *adjMatrix = (float*)calloc(edgeIndexSize*edgeIndexSize, sizeof(float));

	std::cout << "DEBUG: coo2sparse operation start...\n";
        coo2sparse(edgeIndex, adjMatrix, edgeIndexSize, numOfNodes);
	std::cout << "DEBUG: coo2sparse operation successful!\n";

	/*for(int i=0; i<numOfNodes; i++) {
                for(int j=0; j<numOfNodes; j++) {
                        std::cout << *(adjMatrix + i*numOfNodes + j) << " ";
                }
                std::cout << std::endl;
        }*/

	float* outputMatrix = (float*)calloc(numOfNodes * featureSize, sizeof(float));

	std::cout << "DEBUG: CU_SpMM::GCN start...\n";
	start = std::chrono::steady_clock::now();
	CU_SpMM::GCNLayer(adjMatrix, h_featureVector, numOfNodes, edgeIndexSize, featureSize, outputMatrix);
	end = std::chrono::steady_clock::now();
        dur_ms = end-start;
        std::cout << "1-layer CU_SpMM::GCN execution took " << dur_ms.count() << " ms\n";

	std::cout << "CU_SpMM::GCNLayer operation returned successfully!\n";

	//for(int i=0; i<featureSize; i++) {
	//	for(int j=0; j<numOfNodes; j++) {
	//		std::cout << *(outputMatrix + i*numOfNodes + j) << " ";
	//	}
	//	std::cout << std::endl;
	//}

	cudaFree(edgeIndex);
	cudaFree(featureVector);
	cudaFree(featureVectorOutput);
	cudaFree(aggregationVar);
	cudaFree(nodeDegrees);

	//std::cout << "nodeMap.size(): " << nodeMap.size() << std::endl;
	//for( const std::pair<int,int>& n : nodeMap ) {
	//	std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
	//}

	return 0;
}

int getEdgeIndexSizeFromFile(const char* fileName) {

	std::ifstream dsFile(fileName);
	std::string line;

	int numOfEdges = 0;

	if(dsFile.is_open()) {
		while(getline(dsFile, line)) {
			numOfEdges++;
		}
	}
	else {
		std::cout << "Could not open the edgeIndex dataset file!\n";
		return -1;
	}

	return numOfEdges;

}

void loadEdgeIndexFromFile(const char* fileName, float* edgeIndex, const int numOfEdges,
			   std::unordered_map<int, int> &nodeMap) {

	std::ifstream dsFile(fileName);
	std::string line;

	int i=0, j=0;

	if(dsFile.is_open()) {
		while(getline(dsFile, line)) {
			std::istringstream ss(line);
			std::string word;
			while(std::getline(ss, word, '\t')) {
				*(edgeIndex + (numOfEdges*i) + j) = nodeMap.find(std::stof(word))->second;
				i = 1;
			}
		j++;
		i = 0;
		}
	}
	else {
		std::cout << "Could not open the feature dataset file!\n";
	}

}

int getFeatureSizeFromFile(const char* fileName) {
	std::ifstream dsFile(fileName);
	std::string line, word;

	int numOfFeatures = -1;

	if(dsFile.is_open()) {
		getline(dsFile, line);
		std::istringstream ss(line);
		while(std::getline(ss, word, '\t')) {
			numOfFeatures++;
		}
	}
	else {
		std::cout << "Could not open the feature dataset file!\n";
		return -1;
	}

	return numOfFeatures;
}

int getNumOfNodesFromFile(const char* fileName) {
	std::ifstream dsFile(fileName);
        std::string line;

        int numOfNodes = 0;

        if(dsFile.is_open()) {
                while(getline(dsFile, line)) {
			numOfNodes++;
		}
        }
        else {
                std::cout << "Could not open the feature dataset file!\n";
                return -1;
        }

        return numOfNodes;

}

void loadFeatureVectorFromFile(const char* fileName, float* featureVector, int featureSize, std::unordered_map<int, int> &nodeMap) {
	std::ifstream dsFile(fileName);
        std::string line;

        int i=0, j=0;

        if(dsFile.is_open()) {
                while(getline(dsFile, line)) {
                        std::istringstream ss(line);
                        std::string word;
			std::getline(ss, word, '\t'); // escape node index
			nodeMap.insert({std::stof(word),i});
                        while(std::getline(ss, word, '\t')) {
				if(word.length() < 5) {
                                    *(featureVector + i*featureSize + j) = std::stof(word);
                                    j += 1;
				}
                        }
                i++;
                j = 0;
                }
        }
        else {
                std::cout << "Could not open the feature dataset file!\n";
        }
}


#ifdef __cplusplus
}
#endif
