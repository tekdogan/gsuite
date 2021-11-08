#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<chrono>
#include<cstdlib>
#include"C_GCN_MP.h"
#include"C_GIN_WL.h"
#include"C_GCN_SpMM.h"
#include"Data_Util.h"

int getEdgeIndexSizeFromFile(const char* fileName);

void loadEdgeIndexFromFile(const char* fileName, double* edgeIndex, const int numOfEdges);

int getFeatureSizeFromFile(const char* fileName);

int getNumOfNodesFromFile(const char* fileName);

void loadFeatureVectorFromFile(const char* fileName, double* featureVector, int featureSize);
