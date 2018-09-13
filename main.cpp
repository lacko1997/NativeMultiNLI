#include <CL\cl.h>
#include "kernel_loader.h"
#include "neural_network.h"
#include "list.h"
#include "common.h"
#include <chrono>

using namespace std;

int main() {
	KernelReader loader = KernelReader("machine_learning.cl");
	const char* src = loader.getKernelSource();

	OpenCL *cl_ctx = new OpenCL(src, MATRIX_SIZE_SMALL);
	NeuralNetwork::getKernels(cl_ctx);
	NeuralNetwork *network = new NeuralNetwork(cl_ctx);

	network->setOutput(0,48);
	network->getMemoryInfo();

	return 0;
}
