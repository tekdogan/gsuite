// This function is designed for a temporary use of
// direct usage of CUDA data loader utility from
// a cpp based main file.

#include"DataLoader.h"

int main(int argc, char *argv[]) {

	if(argc == 1) {
		std::cout << "Please pass a parameter to executable. (e.g. ./cudaDataLoader.o 2)\n";
	}

	LoadData(atoi(argv[1]));

	return 0;
}
