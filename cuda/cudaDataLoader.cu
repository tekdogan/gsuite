#include"DataLoader.h"

// number of threads per block
#define TPB 1
#define NUM_OF_THREADS 1024

#ifdef __cplusplus
extern "C" {
#endif

int LoadData(int arg) {


	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );
	float *h_edgeIndex, *h_featureVector;
	float *edgeIndex, *featureVector;

	std::unordered_map<int,int> nodeMap;

	const char* edgeIndexFileName = "cora.cites";
	int edgeIndexSize = getEdgeIndexSizeFromFile(edgeIndexFileName);
	std::cout << "edgeIndexSize: " << edgeIndexSize << std::endl;

	const char* featureFileName = "cora.content";
	int featureSize = getFeatureSizeFromFile(featureFileName);
	std::cout << "featureSize: " << featureSize << std::endl;

	int numOfNodes = getNumOfNodesFromFile(featureFileName);
	std::cout << "numOfNodes: " << numOfNodes << std::endl;

	try {
		h_featureVector = (float*) calloc(numOfNodes*featureSize, sizeof(float));
		loadFeatureVectorFromFile(featureFileName, h_featureVector, featureSize, nodeMap);
		cudaMalloc( (void**) &featureVector, numOfNodes*featureSize * sizeof(float));
		cudaMemcpy(featureVector, h_featureVector, numOfNodes*featureSize * sizeof(float), cudaMemcpyHostToDevice);
	} catch(...) {
		std::cout << "Could not allocate memory space for featureVector!\n";
	}
	
	try {
		h_edgeIndex = (float*) calloc(2*edgeIndexSize, sizeof(float));
		loadEdgeIndexFromFile(edgeIndexFileName, h_edgeIndex, edgeIndexSize, nodeMap);
		cudaMalloc( (void**) &edgeIndex, 2*edgeIndexSize * sizeof(float));
		cudaMemcpy(edgeIndex, h_edgeIndex, (size_t)2*edgeIndexSize*sizeof(float), cudaMemcpyHostToDevice);
		
	} catch(...) {
		std::cout << "Could not allocate memory space for edgeIndex!\n";
	}

        float *h_aggregationVar = (float*)calloc(numOfNodes * featureSize, sizeof(float));
	float *h_nodeDegrees = (float*)calloc(numOfNodes, sizeof(float));
        float *aggregationVar, *nodeDegrees;

	for(int i=0; i<edgeIndexSize; i++) {
		h_nodeDegrees[(int)(*(h_edgeIndex + i))]++;
	}

	for(int i=0; i<numOfNodes*featureSize; i++) {
		*(h_aggregationVar + i) = 0.0;
	}

	cudaMalloc( (void**) &aggregationVar, numOfNodes * featureSize * sizeof(float));
	cudaMemcpy(aggregationVar, h_aggregationVar, numOfNodes * featureSize * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc( (void**) &nodeDegrees, featureSize*sizeof(float));
	cudaMemcpy(nodeDegrees, h_nodeDegrees, featureSize*sizeof(float), cudaMemcpyHostToDevice);


	int *arr = (int*)calloc(2*edgeIndexSize, sizeof(int));

	/*for(int i=0; i<8; i++) {
	                //std::cout << "copying: " << (int)( *(h_edgeIndex + 2*i + j)) << std::endl;
	                *(arr + i) = (int)( *(h_edgeIndex + i));
	}

	gpu_qsort(arr, edgeIndexSize, edgeIndexSize);

	for(int i=0; i<8; i++) {
	                //std::cout << "copying: " << (int)( *(h_edgeIndex + 2*i + j)) << std::endl;
	                ( *(h_edgeIndex + i)) = (float)( *(arr + i));
	}*/



	//unordered_map<string, double>:: iterator itr;
	//for (itr = umap.begin(); itr != umap.end(); itr++)

	if(arg == 0) { // execute CU_MP_GCN

	auto start = std::chrono::steady_clock::now();

	cudaProfilerStart();

	CU_MP::GCNLayerNew<<<numOfNodes,512>>>(edgeIndex, featureVector, aggregationVar, nodeDegrees, numOfNodes, featureSize, edgeIndexSize);

	//CU_MP::GCNLayerNew<<<16,SIZE>>>(edgeIndex, featureVector, aggregationVar, nodeDegrees, numOfNodes, featureSize, edgeIndexSize);

	cudaProfilerStop();

	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double, std::milli> dur_ms = end-start;
	std::cout << "2-layer GCN execution took " << dur_ms.count() << " ms\n";

	//for(int i=0; i<3; i++) {
	//	std::cout << "Node" << i << " feature 0: " << *(featureVector + featureSize*i) << std::endl;
	//	std::cout << "Node" << i << " feature 1: " << *(featureVector + featureSize*i + 1) << std::endl;
	//}

	} // CU_MP_GCN end

	else if(arg == 1) { // execute CU_SpMM::GIN

	float *featureVectorOutput;
	cudaMalloc( (void**) &featureVectorOutput, numOfNodes*featureSize * sizeof(float));

	float *adjMatrix = (float*)calloc(edgeIndexSize*edgeIndexSize, sizeof(float));
	std::cout << "DEBUG: coo2sparse operation start...\n";
	coo2sparse(h_edgeIndex, adjMatrix, edgeIndexSize, numOfNodes);
	std::cout << "DEBUG: coo2sparse operation successful!\n";

	float* outputMatrix = (float*)calloc(numOfNodes * featureSize, sizeof(float));

        auto start = std::chrono::steady_clock::now();
	cudaProfilerStart();
        CU_SpMM::GINLayer(adjMatrix, h_featureVector, numOfNodes, edgeIndexSize, featureSize, outputMatrix, 0.1);
	cudaProfilerStop();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> dur_ms = end-start;
        std::cout << "1-layer GIN execution took " << dur_ms.count() << " ms\n";

	//for(int i=0; i<3; i++) {
        //        std::cout << "Node" << i << " feature 0: " << *(featureVectorOutput + featureSize*i) << std::endl;
        //        std::cout << "Node" << i << " feature 1: " << *(featureVectorOutput + featureSize*i + 1) << std::endl;
        //}

	} // CU_SpMM::GIN end

	else if(arg == 2) { // execute CU_SpMM_GCN

	//float *adjMatrix = (float*)calloc(numOfNodes*numOfNodes, sizeof(float));
	float *adjMatrix = (float*)calloc(edgeIndexSize*edgeIndexSize, sizeof(float));

	std::cout << "DEBUG: coo2sparse operation start...\n";
        coo2sparse(h_edgeIndex, adjMatrix, edgeIndexSize, numOfNodes);
	std::cout << "DEBUG: coo2sparse operation successful!\n";

	/*for(int i=0; i<numOfNodes; i++) {
                for(int j=0; j<numOfNodes; j++) {
                        std::cout << *(adjMatrix + i*numOfNodes + j) << " ";
                }
                std::cout << std::endl;
        }*/

	float* outputMatrix = (float*)calloc(numOfNodes * featureSize, sizeof(float));

	std::cout << "DEBUG: CU_SpMM::GCN start...\n";
	auto start = std::chrono::steady_clock::now();
	cudaProfilerStart();
	CU_SpMM::GCNLayer(adjMatrix, h_featureVector, numOfNodes, edgeIndexSize, featureSize, outputMatrix);
	cudaProfilerStop();
	auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> dur_ms = end-start;
        std::cout << "1-layer CU_SpMM::GCN execution took " << dur_ms.count() << " ms\n";

	std::cout << "CU_SpMM::GCNLayer operation returned successfully!\n";

	//for(int i=0; i<featureSize; i++) {
	//	for(int j=0; j<numOfNodes; j++) {
	//		std::cout << *(outputMatrix + i*numOfNodes + j) << " ";
	//	}
	//	std::cout << std::endl;
	//}

	} // CU_SpMM_GCN end

	else if(arg == 3) { // execute CU_WL::SAG

	float* outputFeatureMatrix, *tempFeatureValues;
	cudaMalloc(&outputFeatureMatrix, featureSize * numOfNodes * sizeof(float));	
	cudaMalloc(&tempFeatureValues, featureSize * numOfNodes * sizeof(float));

	auto start = std::chrono::steady_clock::now();
	cudaProfilerStart();
	//CU_WL::SAGLayer<<<numOfNodes,512>>>(edgeIndex, featureVector, 1.0, 0.2, numOfNodes, edgeIndexSize, featureSize, tempFeatureValues, outputFeatureMatrix);
	//CU_WL::SAGLayer2<<<numOfNodes/TPB,TPB>>>(edgeIndex, featureVector, 1.0, 0.2, numOfNodes, edgeIndexSize, featureSize, tempFeatureValues, outputFeatureMatrix);
	cudaProfilerStop();
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double, std::milli> dur_ms = end-start;
	std::cout << "1-layer CU_WL::SAG execution took " << dur_ms.count() << " ms\n";

	float* h_outputFeatureMatrix = (float*)calloc(featureSize * numOfNodes, sizeof(float));
	cudaMemcpy(h_outputFeatureMatrix, outputFeatureMatrix, featureSize * numOfNodes * sizeof(float), cudaMemcpyDeviceToHost);

	//std::cout << "Output feature matrix:\n";
	//printDenseMatrix(h_outputFeatureMatrix, featureSize, numOfNodes);

	} // CU_WL::SAG end

	else if(arg == 4) { // execute CU_WL::GIN

        float* outputFeatureMatrix, *tempFeatureValues;
        cudaMalloc(&outputFeatureMatrix, featureSize * numOfNodes * sizeof(float));
        cudaMalloc(&tempFeatureValues, featureSize * numOfNodes * sizeof(float));

        auto start = std::chrono::steady_clock::now();
        cudaProfilerStart();
        CU_WL::GINLayer<<<numOfNodes*featureSize*edgeIndexSize,NUM_OF_THREADS>>>(edgeIndex, featureVector, tempFeatureValues, 0.3, numOfNodes, edgeIndexSize, featureSize, outputFeatureMatrix);
        cudaProfilerStop();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> dur_ms = end-start;
        std::cout << "1-layer CU_WL::GIN execution took " << dur_ms.count() << " ms\n";

        float* h_outputFeatureMatrix = (float*)calloc(featureSize * numOfNodes, sizeof(float));
        cudaMemcpy(h_outputFeatureMatrix, outputFeatureMatrix, featureSize * numOfNodes * sizeof(float), cudaMemcpyDeviceToHost);

	}

	//std::cout << "edge index matrix before sort:\n";
	//printDenseMatrix(h_edgeIndex, 2, edgeIndexSize);
	//std::cout << std::endl;

	//for(int i=0; i<edgeIndexSize*2; i++)
	//	std::cout << "#" << *(h_edgeIndex+i);
	//std::cout << std::endl;

	//std::cout << "feature matrix:\n";
	//printDenseMatrix(h_featureVector, featureSize, numOfNodes);
	//std::cout << std::endl;

	cudaFree(edgeIndex);
	cudaFree(featureVector);
	//cudaFree(featureVectorOutput);
	cudaFree(aggregationVar);
	cudaFree(nodeDegrees);

	//std::cout << "nodeMap.size(): " << nodeMap.size() << std::endl;
	//for( const std::pair<int,int>& n : nodeMap ) {
	//	std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
	//}

	/*int *arr = (int*)calloc(2*edgeIndexSize, sizeof(int));

	for(int i=0; i<8; i++) {
			//std::cout << "copying: " << (int)( *(h_edgeIndex + 2*i + j)) << std::endl;
			*(arr + i) = (int)( *(h_edgeIndex + i));
	}

	gpu_qsort(arr, 4, edgeIndexSize);

	for(int i=0; i<8; i++) {
	                //std::cout << "copying: " << (int)( *(h_edgeIndex + 2*i + j)) << std::endl;
	                ( *(h_edgeIndex + i)) = (float)( *(arr + i));
	}*/

	//std::cout << "edge index matrix after sort:\n";
	//printDenseMatrix(h_edgeIndex, 2, edgeIndexSize);
	//std::cout << std::endl;
	

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
		std::cout << "DEBUG: edgeIndex loaded! j=" << j << std::endl;
	}
	else {
		std::cout << "Could not open the feature dataset file!\n";
	}

}

void loadEdgeIndexFromFile2(const char* fileName, float* edgeIndex, const int numOfEdges,
                           std::unordered_map<int, int> &nodeMap) {

        std::ifstream dsFile(fileName);
        std::string line;

        int i=0, j=0;

        if(dsFile.is_open()) {
                while(getline(dsFile, line)) {
                        std::istringstream ss(line);
                        std::string word;
                        while(std::getline(ss, word, '\t')) {
                                *(edgeIndex + i + j) = nodeMap.find(std::stof(word))->second;
                                i = 1;
                        }
                j++;
                i = 0;
                }
                std::cout << "DEBUG: edgeIndex loaded! j=" << j << std::endl;
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
