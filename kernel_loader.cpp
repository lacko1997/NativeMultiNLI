#include "kernel_loader.h"

KernelReader::KernelReader(const char * filename){
	ifstream file(filename);
	string line;
	while (getline(file, line)) {
		source += line + "\n";
	}
}
