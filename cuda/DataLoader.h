#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<chrono>
#include<cstdlib>
#include"C_GCN_MP.h"
#include"CU_GIN_WL.h"
#include<cuda.h>
#include<cublas_v2.h>
#include"CU_SpMM_GCN.h"
#include"CU_SpMM_GIN.h"
#include"CU_SAG_WL.h"
#include"Data_Util.h"
#include<unordered_map>
#include<cuda_profiler_api.h>
#include"sort/cuda_sort.h"

extern "C" {

// for direct usage from cpp main
int LoadData(int);

int getEdgeIndexSizeFromFile(const char* fileName);

void loadEdgeIndexFromFile(const char* fileName, float* edgeIndex, const int numOfEdges, std::unordered_map<int, int> &nodeMap);

void loadEdgeIndexFromFile2(const char* fileName, float* edgeIndex, const int numOfEdges, std::unordered_map<int, int> &nodeMap);

int getFeatureSizeFromFile(const char* fileName);

int getNumOfNodesFromFile(const char* fileName);

void loadFeatureVectorFromFile(const char* fileName, float* featureVector, int featureSize, std::unordered_map<int, int> &nodeMap);

}
