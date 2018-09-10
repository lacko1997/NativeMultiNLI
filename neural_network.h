#pragma once
#include <vector>
#include <random>
#include "opencl_class.h"
#include "ptr_list.h"
#include "common.h"


typedef struct graph_point;
typedef struct connection;
typedef struct connection {
	uint32_t id;
	cl_mem mat_mem;
	cl_mem bias_mem;
	
	fvector biases;
	matrix connection_weights;

	cl_kernel activation;

	graph_point* from;
	graph_point* to;

	bool operator<(connection other) { return other.id > id; }
	bool operator>(connection other) { return other.id < id; }
	bool operator==(connection other) { return other.id == id; }
}connection;

typedef struct ClasssifiedTrainingInput {
	float **input;
	uint32_t type;
}ClassifiedTrainigInput;

typedef struct RecurrentClassifiedTrainingInput {
	uint32_t vec_count;
	float ***input;
	uint32_t type;
}RecurrentClassifiedTrainingInput;

typedef struct graph_point {
	uint32_t id;
	bool visited = false;
	bool finished = false;

	uint32_t kernel_layer_size;
	uint32_t layer_size;

	cl_mem layer_mem;

	Ptr_List<connection*> *out;
	Ptr_List<connection*> *in;

	bool operator<(graph_point other) { return other.id > id; }
	bool operator>(graph_point other) { return other.id < id; }
	bool operator==(graph_point other) { return other.id == id; }
};
class NeuralNetwork {
private:
	Ptr_List<graph_point*> *input;
	graph_point *output = NULL;

	vector<graph_point*> *graph_points;
	vector<connection*> *connections;
	OpenCL *context;

	cl_kernel reduce_sum;
	cl_kernel softmax_pow;
	cl_kernel skalar_div;
	cl_kernel vec_mat_mul;
	cl_kernel vec_mat_mul_add;

	float* output_data;
	cl_mem output_mem;

	void softmax();
	bool find_graph_point(graph_point *index,uint32_t *loc);
	bool insert_graph_point(graph_point *index);

	bool find_connection(connection *index, uint32_t *loc);
	bool insert_connection(connection *conn);

	void init();
	void copy_to_input(float** data);
	void forward_propagation(float* data);
public:
	NeuralNetwork(OpenCL *context);
	void trainRecurrent(uint32_t input_count,RecurrentClassifiedTrainingInput *inputs);
	void train(float** inputs,uint32_t type);
	uint32_t predict(float **inputs);

	void connectLayers(uint32_t src, uint32_t dst,uint32_t conn_id, cl_kernel activation);
	bool findGraphPointById(uint32_t id, uint32_t *loc);
	bool findConnectionById(uint32_t id, uint32_t *loc);
	void setOutput(uint32_t layer_id,uint32_t layer_size);

	void addInputLayer(uint32_t layer_id, uint32_t layer_size);
	void addLayer(uint32_t layer_id, uint32_t layer_size, cl_kernel activation);
};