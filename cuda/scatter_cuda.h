#pragma once

#include <iostream>

float* scatter_cuda(float *src, float *index, int64_t dim,
             std::string reduce, int numOfNodes, int numOfFeatures,
             int numOfEdges);
