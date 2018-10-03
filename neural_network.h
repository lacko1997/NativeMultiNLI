#pragma once
#include <vector>
#include <random>
#include "opencl_class.h"
#include "ptr_set.h"
#include "ptr_list.h"
#include "common.h"
#include "graph_point.h"

class NeuralNetwork {
private:
	static cl_kernel reduce_sum;
	static cl_kernel softmax_pow;
	static cl_kernel skalar_div;
	static cl_kernel vec_mat_mul;
	static cl_kernel vec_mat_mul_add;
	static cl_kernel add;
	static cl_kernel cross_entropy;

	Ptr_List<graph_point*> *input;
	graph_point *output = NULL;

	vector<graph_point*> *graph_points;
	vector<connection*> *connections;
	OpenCL *context;

	float* result_data;
	cl_mem result_mem;
	float loss_value;
	int32_t last_index;

	void softmax();
	bool find_graph_point(graph_point *index,uint32_t *loc);
	bool insert_graph_point(graph_point *index);

	bool find_connection(connection *index, uint32_t *loc);
	bool insert_connection(connection *conn);

	void init();
	void copy_to_input(float** data);
	void forward_propagation(float* data);
	void loss(uint32_t index);
	void back_propagation(uint32_t index);
public:
	static void getKernels(OpenCL *context);
	static void releaseKernels(OpenCL *context);

	NeuralNetwork(OpenCL *context);
	~NeuralNetwork();
	void trainRecurrent(uint32_t input_count,RecurrentClassifiedTrainingInput *inputs);
	void train(uint32_t input_count,ClassifiedTrainigInput *inputs);
	uint32_t predict(float **inputs);

	void getMemoryInfo();
	void connectLayers(uint32_t src, uint32_t dst,uint32_t conn_id, cl_kernel *activation);
	bool findGraphPointById(uint32_t id, uint32_t *loc);
	bool findConnectionById(uint32_t id, uint32_t *loc);
	void setOutput(uint32_t layer_id,uint32_t layer_size);

	void addInputLayer(uint32_t layer_id, uint32_t layer_size);
	void addLayer(uint32_t layer_id, uint32_t layer_size, NNLayerType type, cl_kernel activation);
};