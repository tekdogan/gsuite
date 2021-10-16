#include"DataLoader.h"

#define SIZE 1024

int main() {

	float *edgeIndex, *featureVector;

	float aggregationVar[2] = {0,0};
	float nodeDegrees[3] = {1,2,1};

	const char* edgeIndexFileName = "cora.cites";
	int edgeIndexSize = getEdgeIndexSizeFromFile(edgeIndexFileName);
	std::cout << "edgeIndexSize: " << edgeIndexSize << std::endl;

	const char* featureFileName = "cora.content";
	int featureSize = getFeatureSizeFromFile(featureFileName);
	std::cout << "featureSize: " << featureSize << std::endl;

	int numOfNodes = getNumOfNodesFromFile(featureFileName);
	std::cout << "numOfNodes: " << numOfNodes << std::endl;

	try {
		cudaMallocManaged( (void**) &edgeIndex, (unsigned long) featureSize*edgeIndexSize * sizeof(float));
		loadEdgeIndexFromFile(edgeIndexFileName, edgeIndex, edgeIndexSize);
	} catch(...) {
		std::cout << "Could not allocate memory space for edgeIndex!\n";
	}

	try {
		cudaMallocManaged( (void**) &featureVector, numOfNodes*featureSize * sizeof(float));
		loadFeatureVectorFromFile(featureFileName, featureVector, featureSize);
	} catch(...) {
		std::cout << "Could not allocate memory space for featureVector!\n";
	}

	auto start = std::chrono::steady_clock::now();

	CU_MP::GCNLayerNew<<<16,SIZE>>>(edgeIndex, featureVector, aggregationVar, nodeDegrees);

	CU_MP::GCNLayerNew<<<16,SIZE>>>(edgeIndex, featureVector, aggregationVar, nodeDegrees);

	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double, std::milli> dur_ms = end-start;
	std::cout << "2-layer GCN execution took " << dur_ms.count() << " ms\n";

	for(int i=0; i<3; i++) {
		std::cout << "Node" << i << " feature 0: " << *(featureVector + featureSize*i) << std::endl;
		std::cout << "Node" << i << " feature 1: " << *(featureVector + featureSize*i + 1) << std::endl;
	}


	float *featureVectorOutput;
	cudaMallocManaged( (void**) &featureVectorOutput, numOfNodes*featureSize * sizeof(float));
        start = std::chrono::steady_clock::now();
        //GINLayer<<<16,SIZE>>>(edgeIndex, featureVector, aggregationVar, nodeDegrees, 1.0, featureVectorOutput);
        end = std::chrono::steady_clock::now();
        dur_ms = end-start;
        std::cout << "1-layer GIN execution took " << dur_ms.count() << " ms\n";

	for(int i=0; i<3; i++) {
                std::cout << "Node" << i << " feature 0: " << *(featureVectorOutput + featureSize*i) << std::endl;
                std::cout << "Node" << i << " feature 1: " << *(featureVectorOutput + featureSize*i + 1) << std::endl;
        }

	cudaFree(edgeIndex);
	cudaFree(featureVector);
	cudaFree(featureVectorOutput);

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

void loadEdgeIndexFromFile(const char* fileName, float* edgeIndex, const int numOfEdges) {

	std::ifstream dsFile(fileName);
	std::string line;

	int i=0, j=0;

	if(dsFile.is_open()) {
		while(getline(dsFile, line)) {
			std::istringstream ss(line);
			std::string word;
			while(std::getline(ss, word, ' ')) {
				*(edgeIndex + (numOfEdges*i) + j) = std::stof(word);
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
		while(std::getline(ss, word, ' ')) {
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

void loadFeatureVectorFromFile(const char* fileName, float* featureVector, int featureSize) {
	std::ifstream dsFile(fileName);
        std::string line;

        int i=0, j=0;

        if(dsFile.is_open()) {
                while(getline(dsFile, line)) {
                        std::istringstream ss(line);
                        std::string word;
			std::getline(ss, word, ' '); // escape node index
                        while(std::getline(ss, word, ' ')) {
                                *(featureVector + i*featureSize + j) = std::stof(word);
                                j += 1;
                        }
                i++;
                j = 0;
                }
        }
        else {
                std::cout << "Could not open the feature dataset file!\n";
        }
}
