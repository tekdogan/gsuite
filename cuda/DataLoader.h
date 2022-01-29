#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<chrono>
#include<cstdlib>
#include<cuda.h>
#include<cublas_v2.h>
#include"Data_Util.h"
#include<unordered_map>
#include<cuda_profiler_api.h>
#include"sort/cuda_sort.h"

extern "C" {

// for direct usage from cpp main
int LoadData(int);

int getEdgeIndexSizeFromFile(const char* fileName);

void loadEdgeIndexFromFile(const char* fileName, float* edgeIndex, const int numOfEdges, std::unordered_map<std::string, std::string> &nodeMap);

int getFeatureSizeFromFile(const char* fileName);

int getNumOfNodesFromFile(const char* fileName);

void loadFeatureVectorFromFile(const char* fileName, float* featureVector, int featureSize, std::unordered_map<std::string, std::string> &nodeMap);

}
