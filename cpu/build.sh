#!/bin/bash
echo "Building the below files:"
echo "C_GCN_MP.cpp"
echo "C_GIN_WL.cpp"
echo "DataLoader.cpp"
echo "Data_Util.cpp"

g++ -fopenmp -std=c++0x C_GCN_MP.cpp C_GCN_SpMM.cpp C_GIN_WL.cpp DataLoader.cpp Data_Util.cpp -o DataLoader.o

echo "Build is successful! Executable DataLoader.o is generated."
