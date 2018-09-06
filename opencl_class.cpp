#include "opencl_class.h"


void OpenCL::getDeviceInfo(cl_device_id gpu) {
	cl_ulong local_mem_size;
	clGetDeviceInfo(gpu, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
	int fcount = local_mem_size / sizeof(float);
	cout << fcount << " floats" << endl;

	cl_ulong global_mem_size;
	clGetDeviceInfo(gpu, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
	cout << global_mem_size / (1024 * 1024) << " MB" << endl;
}
bool OpenCL::init_OpenCL() {
	uint32_t platform_c;
	clGetPlatformIDs(0, NULL, &platform_c);
	if (platform_c == 0) {
		cout << "No OpenCL platform found." << endl;//return false;
		return false;
	}
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*platform_c);
	clGetPlatformIDs(platform_c, platforms, &platform_c);

	uint32_t gpu_c;
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &gpu_c);
	gpus = (cl_device_id*)malloc(sizeof(cl_device_id)*gpu_c);
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, gpu_c, gpus, &gpu_c);

	ctx = clCreateContext(NULL, 1, &gpus[0], NULL, NULL, NULL);
	return true;
}

void OpenCL::create_neural_networks_program(const char * src, matrix_size_category category) {
	const char* options=NULL;
	switch (category) {
	case MATRIX_SIZE_SMALL:
		tile_size = 32;
		options = "-D TS=32 -D WPT=8";
		break;
	case MATRIX_SIZE_MEDIUM:
		tile_size = 64;
		options = "-D TS=64 -D WPT=8";
		break;
	case MATRIX_SIZE_LARGE:
		tile_size = 128;
		options = "-D TS=128 -D WPT=16";
		break;
	case MATRIX_SIZE_VERY_LARGE:
		tile_size = 256;
		options = "-D TS=256 -D WPT=16";
		break;
	case MATRIX_SIZE_GIGANTIC:
		tile_size = 512;
		options = "-D TS=512 -D WPT=32";
		break;
	}
	matrix_ops = clCreateProgramWithSource(ctx, 1, &src, NULL, NULL);
	clBuildProgram(matrix_ops, 1, gpus, options, NULL, NULL);
	int32_t res;
	clGetProgramBuildInfo(matrix_ops, gpus[0], CL_PROGRAM_BUILD_STATUS, sizeof(int32_t), &res, 0);
	if (res) {
		uint32_t log_sz;
		clGetProgramBuildInfo(matrix_ops, gpus[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
		char* log = (char*)malloc(log_sz);
		clGetProgramBuildInfo(matrix_ops, gpus[0], CL_PROGRAM_BUILD_LOG, log_sz, log, &log_sz);
		cout << log << endl;
	}
	queue = clCreateCommandQueue(ctx, gpus[0], NULL, NULL);
}

OpenCL::OpenCL(const char* src,matrix_size_category category) {
	success=init_OpenCL();
	if (success) {
		create_neural_networks_program(src, category);
	}
}

matrix matrix::operator=(matrix mat){
	matrix res;
	res.width = mat.width;
	res.height = mat.height;
	res.kernel_width = mat.kernel_width;
	res.data = (float*)malloc(sizeof(float)*mat.kernel_width*mat.height);
	return res;
}