#pragma once
#include <CL/opencl.h>
#include <iostream>

using namespace std;

struct matrix {
	int width;			//The width of the matrix.
	int height;
	int kernel_width;	//An increased value of the matrix. This must be done, because the the width value must be a multiple of a given value.The value depends on the GPU hardware.
	float* data;		//This contains the values of the matrix. If the matrix has an N width smaller then the kernel_width, than the rows Nth row, and the rows after it will contain only zeros.

	matrix operator=(matrix mat);
};
typedef struct fvector {
	int length;
	int kernel_length;
	float *data;
}fvector;

typedef enum matrxi_size_category {
	MATRIX_SIZE_SMALL = 0,
	MATRIX_SIZE_MEDIUM,
	MATRIX_SIZE_LARGE,
	MATRIX_SIZE_VERY_LARGE,
	MATRIX_SIZE_GIGANTIC
}matrix_size_category;

class OpenCL {
private:
	cl_platform_id *platforms;
	cl_device_id *gpus;
	cl_context ctx;
	cl_command_queue queue;
	cl_program matrix_ops;

	uint32_t tile_size;
public:
	void getDeviceInfo(cl_device_id gpu);
	void initOpenCL();
	void createNeuralNetworksProgram(const char * src, matrix_size_category category);

	OpenCL(const char* src, matrix_size_category category);

	cl_context getContext() { return ctx; }
	cl_program getProgram() { return matrix_ops; }
	cl_command_queue getQueue() { return queue; }
	uint32_t getTileSize() { return tile_size; }
};
/*
clEnqueueWriteBuffer(queue, first, CL_FALSE, 0, 5 * sizeof(float), A, 0, NULL, NULL);
clEnqueueWriteBuffer(queue, second, CL_FALSE, 0, 5 * sizeof(float), B, 0, NULL, NULL);

clSetKernelArg(kernel, 0, sizeof(int), &M);
clSetKernelArg(kernel, 1, sizeof(int), &N);
clSetKernelArg(kernel, 2, sizeof(int), &K);
clSetKernelArg(kernel, 3, sizeof(first), &first);
clSetKernelArg(kernel, 4, sizeof(second), &second);
clSetKernelArg(kernel, 5, sizeof(result), &result);

first = clCreateBuffer(ctx, CL_MEM_READ_WRITE,   * sizeof(float), NULL, NULL);
second = clCreateBuffer(ctx, CL_MEM_READ_WRITE,  * sizeof(float), NULL, NULL);
result = clCreateBuffer(ctx, CL_MEM_READ_WRITE,  * sizeof(float), NULL, NULL);
uint32_t global_size = 5;
clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
clEnqueueReadBuffer(queue, result, CL_FALSE, 0, 5 * sizeof(float), C, 0, NULL, NULL);
clFinish(queue);
*/