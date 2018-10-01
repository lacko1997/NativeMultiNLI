#pragma once
struct dense_layer;
struct connection;
struct layer_op;
struct graph_point;

typedef enum NNLayerType {
	NN_LAYER_TYPE_DENSE_LAYER = 0x01,
	NN_LAYER_TYPE_RECURRENT_LAYER = 0x02,
	NN_LAYER_TYPE_LAYER_OP = 0x04
};

typedef struct connection {
	uint32_t id;
	bool visited;

	fvector biases;
	matrix connection_weights;

	cl_kernel activation;
	cl_mem mat_mem;
	cl_mem bias_mem;

	graph_point* from;
	graph_point* to;

	bool operator<(connection other) { return other.id > id; }
	bool operator>(connection other) { return other.id < id; }
	bool operator==(connection other) { return other.id == id; }
}connection;

/*
	The number of the class, where the given input belongs.
	e.g.: the vector describes a boy (type=0), a girl(type=1) or a child=2.
*/
typedef struct ClasssifiedTrainingInput {
	float *input;
	uint32_t type;
}ClassifiedTrainigInput;

/*
   We use this,if the neural network has a recurrent component.
   The type is used for the same reason as in the ClassifiedTrainingInput.
*/
typedef struct RecurrentClassifiedTrainingInput {
	uint32_t vec_count;
	float **input;
	uint32_t type;
}RecurrentClassifiedTrainingInput;

typedef struct dense_layer {
	uint32_t kernel_layer_size;
	int32_t layer_size;
	cl_mem layer_mem;
	cl_mem backprop_mem;

	Ptr_List<connection*> *out;
	Ptr_List<connection*> *in;
}dense_layer;

typedef struct layer_op {
	dense_layer *output;
	Ptr_List<dense_layer*> *inputs;
	dense_layer *(*operation)(Ptr_List<dense_layer>);
}layer_op;

typedef struct graph_point {
	uint8_t type;
	// every layer type must know these values.
	uint32_t id;
	bool visited;

	union {
		dense_layer layer;
		//TODO: recurrent_layer recurrent; uint16_t recurrent;
		layer_op operation;
	};
	//These operators help needed to sort values by their id;
	bool operator<(graph_point other) { return other.id > id; }
	bool operator>(graph_point other) { return other.id < id; }
	bool operator==(graph_point other) { return other.id == id; }

	graph_point& operator=(dense_layer& layer);
	graph_point& operator=(layer_op& operation);
	//We have to be sure that, we assign a graph_point with the same type;
	graph_point& operator=(graph_point point);
}graph_point;
