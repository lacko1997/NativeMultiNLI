#include <CL\cl.h>
#include "kernel_loader.h"
#include "neural_network.h"
#include "common.h"
#include <chrono>

using namespace std;

typedef struct abc{
	uint32_t type;
	union {
		uint32_t ui;
		float sf;
	};
}abc;


int main(int argc,char** argv,char** env) {
	KernelReader loader = KernelReader("machine_learning.cl");
	const char* src = loader.getKernelSource();

	OpenCL *cl_ctx = new OpenCL(src, MATRIX_SIZE_SMALL);

	NeuralNetwork::getKernels(cl_ctx);
	NeuralNetwork *network = new NeuralNetwork(cl_ctx);
	network->addInputLayer(0, 128);
	network->setOutput(1,48);

	network->getMemoryInfo();

	if (argc > 1 && !strcmp(argv[1], "-pause")) {
		system("pause");
	}
	return 0;
}
