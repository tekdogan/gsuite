#include"Data_Util.h"


void matrix_mul(double* m1, double* m2, double* m3, int m1r, int m1c, int m2r, int m2c) {

	int i,j,k;

	for(i=0; i<m1r; i++){
		for(j=0; j<m2c; j++) {
			*(m3 + i*m2c + j) = 0; // init m3 cell
			for(k=0; k<m1c; k++) {
				*(m3 + i*m2c + j) +=
					*(m1 + i*m1c + k) *
					*(m2 + k*m2c + j);
			}
		}
	}
}

void addSelfLoops(double* edgeIndex, int n_edges) {

	int temp_it = n_edges;

	for(int i=0; i<n_edges; i++) {
		*(edgeIndex + temp_it) = 0;
	}

}

void coo2csr(double* csr_row, double* csr_col, double* coo_edge_index, int nnz, int nnodes) {

	for(int i=0; i<nnz; i++) {
		*(csr_col + i) = *(coo_edge_index + nnz + i);
		*(csr_row + ((int)*(coo_edge_index + i)) + 1) += 1.0;
	}

	for(int i=0; i<nnodes; i++) {
		*(csr_row + i + 1) = *(csr_row + i);
	}

}

void csr2coo(double* csr_row, double* csr_col, double* coo_edge_index, int nnz, int nnodes) {

	// TO DO

}

void coo2sparse(float* coo_edge_index, float* adj_matrix, int nnz, int nnodes) {

	for(int i=0; i<nnz; i++) {
		*(adj_matrix + (int)*(coo_edge_index+i)*nnz + (int)*(coo_edge_index + nnz + i)) = 1.0;
	}

}

void printDenseMatrix(float* A, int row, int col) {

	for(int i=0; i<row; i++) {
		for(int j=0; j<col; j++) {
			std::cout << *(A + i*col + j) << " ";
		}
		std::cout << std::endl;
	}

}

