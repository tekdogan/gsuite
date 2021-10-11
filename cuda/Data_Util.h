#include<iostream>

// convert from COO format to CSR format
void coo2csr(double* csr_row, double* csr_col, double* coo_edge_index, int nnz, int nnodes);

// convert fro CSR format to COO format
void csr2coo(double* csr_row, double* csr_col, double* coo_edge_index, int nnz);

// convert from COO format to Sparse Matrix
void coo2sparse(float* coo_edge_index, float* adj_matrix, int nnz, int nnodes);

// method for matrix multiplication
void matrix_mul(double* m1, double* m2, double* m3, int m1r, int m1c, int m2r, int m2c);

// method for printing dense matrices
void printDenseMatrix(float* A, int row, int col);
