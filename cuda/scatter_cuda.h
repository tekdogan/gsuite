#pragma once

#include <tuple>
#include <iostream>

std::tuple<float*, float*>
scatter_cuda(float *src, float *index, int64_t dim,
             std::string reduce, int numOfNodes, int numOfFeatures,
             int numOfEdges);
