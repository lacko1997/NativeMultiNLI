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
	cl_mem memory;
	fvector biases;
	cl_kernel activation;
	matrix connection_weights;
	graph_point* from;
	graph_point* to;

	bool operator<(connection other) { return other.id > id; }
	bool operator>(connection other) { return other.id < id; }
	bool operator==(connection other) { return other.id == id; }
}connection;
typedef struct graph_point {
	uint32_t id;

	uint8_t visited = false;
	uint32_t kernel_layer_size;
	uint32_t layer_size;
	Ptr_List<connection*> *out;
	Ptr_List<connection*> *in;
	
	uint32_t max_outgoing_count();

	bool operator<(graph_point other) { return other.id > id; }
	bool operator>(graph_point other) { return other.id < id; }
	bool operator==(graph_point other) { return other.id == id; }
};
class NeuralNetwork {
private:
	float** layer_data;
	float* output_data;

	cl_mem data_mem;
	cl_mem output_mem;

	Ptr_List<graph_point*> *input;
	graph_point *output = NULL;

	vector<graph_point*> *graph_points;
	vector<connection*> *connections;
	OpenCL *context;

	cl_kernel reduce_sum;
	cl_kernel softmax_pow;
	cl_kernel skalar_div;
	cl_kernel vec_mat_mul;

	void softmax();
	bool find_graph_point(graph_point *index,uint32_t *loc);
	bool insert_graph_point(graph_point *index);

	bool find_connection(connection *index, uint32_t *loc);
	bool insert_connection(connection *conn);

	void init();
	void forward_propagation(float* data);
public:
	NeuralNetwork(OpenCL *context);
	void trainRecurrent(float ***input,uint32_t type);
	void train(float** inputs,uint32_t type);
	uint32_t predict(float **inputs);

	void connectLayers(uint32_t src, uint32_t dst,uint32_t conn_id);
	bool findGraphPointById(uint32_t id, uint32_t *loc);
	bool findConnectionById(uint32_t id, uint32_t *loc);
	void setOutput(uint32_t layer_id,uint32_t layer_size);

	void addInputLayer(uint32_t layer_id, uint32_t layer_size);
	void addLayer(uint32_t layer_id, uint32_t layer_size, cl_kernel activation);
};