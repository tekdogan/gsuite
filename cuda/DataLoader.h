#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<chrono>
#include<cstdlib>
#include"C_GCN_MP.h"
#include"C_GIN_WL.h"
#include<cuda.h>
#include"CU_SpMM_GCN.h"
extern "C" {
#include"Data_Util.h"
}
#include<unordered_map>

int getEdgeIndexSizeFromFile(const char* fileName);

void loadEdgeIndexFromFile(const char* fileName, float* edgeIndex, const int numOfEdges, std::unordered_map<int, int> &nodeMap);

int getFeatureSizeFromFile(const char* fileName);

int getNumOfNodesFromFile(const char* fileName);

void loadFeatureVectorFromFile(const char* fileName, float* featureVector, int featureSize, std::unordered_map<int, int> &nodeMap);
