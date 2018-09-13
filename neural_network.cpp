#include "neural_network.h"

inline void powOfE(OpenCL *context, cl_kernel softmax_pow, cl_mem output_mem, graph_point *output, uint32_t *size, float *output_data) {
	float sum = 0.0f;
	clEnqueueWriteBuffer(context->getQueue(), output_mem, false, 0, size[0] * sizeof(float), output_data, 0, NULL, NULL);
	clSetKernelArg(softmax_pow, 0, sizeof(uint32_t), &output->layer_size);
	clSetKernelArg(softmax_pow, 1, sizeof(output_mem), &output_mem);
	clSetKernelArg(softmax_pow, 2, sizeof(output_mem), &output_mem);
	clEnqueueNDRangeKernel(context->getQueue(), softmax_pow, 1, NULL, size, NULL, 0, NULL, NULL);
}

inline void sum_elements(OpenCL *context,cl_kernel reduce_sum,uint32_t *size,uint32_t *sizel,cl_mem memory,float *sum) {
	bool belowTS = false;
	uint32_t TS = context->getTileSize();
	while (size[0] / 2 >= 1) {
		uint32_t in = size[0];
		clSetKernelArg(reduce_sum, 0, sizeof(uint32_t), &in);
		clSetKernelArg(reduce_sum, 1, sizeof(memory), &memory);
		clSetKernelArg(reduce_sum, 2, sizeof(memory), &memory);
		if (size[0] >= sizel[0]) {
			clEnqueueNDRangeKernel(context->getQueue(), reduce_sum, 1, NULL, size, sizel, 0, NULL, NULL);
		}
		else {
			clEnqueueNDRangeKernel(context->getQueue(), reduce_sum, 1, NULL, sizel, sizel, 0, NULL, NULL);
		}
		size[0] = size[0] / 2;
		if (size[0] % TS != 0 && !belowTS) {
			size[0] += TS - size[0] % TS;
			if (size[0] == TS) {
				belowTS = true;
			}
		}
	}
	clEnqueueReadBuffer(context->getQueue(), memory, false, 0, sizeof(float), sum, 0, NULL, NULL);
	clFinish(context->getQueue());
}

cl_kernel NeuralNetwork::reduce_sum;
cl_kernel NeuralNetwork::softmax_pow;
cl_kernel NeuralNetwork::skalar_div;
cl_kernel NeuralNetwork::vec_mat_mul;
cl_kernel NeuralNetwork::vec_mat_mul_add;

void NeuralNetwork::softmax() {
	size_t size[] = { output->kernel_layer_size };
	size_t sizel[] = { context->getTileSize() };
	//raise the output to the power of E (Euler-number)
	powOfE(context,softmax_pow,output->layer_mem,output,size,output_data);
	float sum[1];
	sum_elements(context, reduce_sum, size, sizel, output_mem, sum);
	size[0] = output->kernel_layer_size;

	clSetKernelArg(skalar_div,0,sizeof(float),sum);
	clSetKernelArg(skalar_div, 1, sizeof(output_mem), &output_mem);
	clSetKernelArg(skalar_div, 2, sizeof(output_mem), &output_mem);
	clEnqueueNDRangeKernel(context->getQueue(), skalar_div, 1, NULL, size, NULL, 0, NULL, NULL);
	float *result = (float*)malloc(sizeof(float)*size[0]);
	clEnqueueReadBuffer(context->getQueue(), output_mem, false, 0, sizeof(float), result, 0, NULL, NULL);
	clFinish(context->getQueue());
	for (int i = 0; i < size[0]; i++) {
		cout << result[i] << " ";
	}
	cout << endl;
}

bool NeuralNetwork::find_graph_point(graph_point *index, uint32_t *loc){
	if (graph_points->size() == 0) {
		*loc = 0;
		return false;
	}
	int S = 0;
	int L = graph_points->size()-1;
	int M = (S + L) / 2;
	while (S <= L) {
		if (*(*graph_points)[M] == *index) {
			*loc = M;
			return true;
		}else if (*(*graph_points)[M] > *index) {
		L = M;
		}else if(*(*graph_points)[M] < *index){
			S = M + 1;
		}
		M = (S + L) / 2;
		if (L == M && M == S) {
			*loc = M;
			return *(*graph_points)[M] == *index;
		}
	}
	*loc = M;
	return false;
}

void NeuralNetwork::getMemoryInfo(){
	uint32_t bytes = sizeof(NeuralNetwork);
	cout << "NeuralNetwork class size: " << sizeof(NeuralNetwork) << endl;
	if (input) {
		uint32_t input_sz = input->size() * sizeof(graph_point*) + sizeof(Ptr_List<graph_point*>);
		bytes += input_sz;
		cout << "The size of the input list in bytes: i" << input_sz << endl;
	}
	if (graph_points) {
		uint32_t graph_point_bytes = sizeof(vector<graph_point*>);
		graph_point_bytes += (sizeof(graph_point*)+sizeof(graph_point))*graph_points->size();
		for (int i = 0; i < graph_points->size(); i++) {
			if ((*graph_points)[i]->in) {
				graph_point_bytes += sizeof(Ptr_List<graph_point*>);
			}
		}
		bytes += graph_point_bytes;
	}
}

void NeuralNetwork::connectLayers(uint32_t src, uint32_t dst,uint32_t conn_id,cl_kernel activation){
	uint32_t loc_src, loc_dst;
	if (findGraphPointById(src, &loc_src)&&findGraphPointById(dst,&loc_dst)) {
		graph_point **input_layer=input->iterator();
		if (output->id == src) {
			cout << "[Source] Graph point with id " << src << "is an output layer. It cannot be the source of a connection." << endl;
			return;
		}
		while (input_layer) {
			if ((*input_layer)->id == dst) {
				cout << "[Destination] Graph point with id " << dst << " is an input layer. It cannot be the destination of a connection." << endl;
				return;
			}
			input_layer=input->next();
		}
		connection *conn = (connection*)malloc(sizeof(float));
		conn->id = conn_id;
		if (!insert_connection(conn)) {
			free(conn);
			cout << "connection with id: " << conn->id << " already exsist.";
			return;
		}
		conn->from = (*graph_points)[loc_src];
		conn->to = (*graph_points)[loc_dst];
		conn->connection_weights.width = conn->to->layer_size;
		conn->connection_weights.height = conn->from->layer_size;

		uint32_t mWidth = conn->to->layer_size;
		uint32_t remainder = mWidth % context->getTileSize();
		if (remainder != 0) {
			mWidth += context->getTileSize() - remainder;
		}
		uint32_t mHeight = (*graph_points)[loc_dst]->layer_size;
		conn->connection_weights.kernel_width = mWidth;
		conn->connection_weights.data = (float*)malloc(sizeof(float)*mWidth*mHeight);

		(*graph_points)[loc_src]->out->push_back(conn);
		(*graph_points)[loc_dst]->in->push_back(conn);
		
		uint32_t width = conn->connection_weights.width;
		uint32_t height = conn->connection_weights.height;
		uint32_t kernel_w= conn->connection_weights.kernel_width;
		for (int y = 0; y <height; y++) {
			for (int x = 0; x < kernel_w; x++) {
				if (!(x < width)) {
					conn->connection_weights.data[y*kernel_w + x]=0.0f;
				}
			}
		}
	}else {
		if (!findGraphPointById(src, &loc_src)) {
			cout << "[Source] Graph point with id " << src << " does not exsist." << endl;
		}
		if (!findGraphPointById(src, &loc_src)) {
			cout << "[Destination] Graph point with id " << dst << " does not exsist." << endl;
		}
	}
}
bool NeuralNetwork::findGraphPointById(uint32_t id, uint32_t *loc) {
	int S = 0;
	int L = graph_points->size() - 1;
	int M = (S + L) / 2;
	while (S <= L) {
		if ((*graph_points)[M]->id == id) {
		*loc = M;
			return true;
		}
		else if ((*graph_points)[M]->id > id) {
			L = M;
		}
		else if ((*graph_points)[M]->id < id) {
			S = M + 1;
		}
		M = (S + L) / 2;
		if (L == M && M == S) {
			*loc = M;
			return (*graph_points)[M]->id == id;
		}
	}
	*loc = M;
	return false;
}

void NeuralNetwork::setOutput(uint32_t layer_id, uint32_t layer_size){
	if (!context->isCreated()) {
		cout << "OpenCL not supported. NeuralNetwork::setOutput cannot be called."<<endl;
		return;
	}
	if (output != NULL) {
		cout << "An output layer was already set." << endl;
		return;
	}
	output = (graph_point*)malloc(sizeof(graph_point));
	output->id = layer_id;
	if (insert_graph_point(output)) {
		output->layer_size=layer_size;
		if (layer_size % 32 != 0) {
			layer_size += (32 - layer_size % 32);
		}
		output->kernel_layer_size = layer_size;
		output->in = new Ptr_List<connection*>();
		output->visited = false;
	}else {
		cout << "A layer with the id: " << layer_id << " already exists." << endl;
		free(output);
	};
	//cout << endl;
	//softmax();
}

bool NeuralNetwork::insert_graph_point(graph_point *index){
	uint32_t loc;
	graph_point point = *index;
	if (graph_points->size() == 0) {
		graph_points->push_back(index);
		return true;
	}
	if (graph_points->size() == 1) {
		if (*(*graph_points)[0] > point) {
			graph_points->insert(graph_points->begin(), index);
		}else {
			graph_points->push_back(index);
		}
		return true;
	}
	if (!find_graph_point(index,&loc)) {
		if (loc == graph_points->size() - 1) {
			if (*(*graph_points)[loc] > point) {
				graph_points->insert(graph_points->begin() + loc, index);
			}else {
				graph_points->push_back(index);
			}
			return true;
		}
		if (*(*graph_points)[loc] > point) {
			graph_points->insert(graph_points->begin() + loc, index);
		}else {
			graph_points->insert(graph_points->begin() + (loc+1), index);
		}
		return true;
	}else {
		return false;
	}
}

void NeuralNetwork::getKernels(OpenCL * context){
	reduce_sum = clCreateKernel(context->getProgram(), "reduce_sum", NULL);
	softmax_pow = clCreateKernel(context->getProgram(), "softmax_pow", NULL);
	skalar_div = clCreateKernel(context->getProgram(), "skalar_div", NULL);
	vec_mat_mul = clCreateKernel(context->getProgram(), "vec_mat_mul", NULL);
	vec_mat_mul_add = clCreateKernel(context->getProgram(), "vec_mat_mul_add", NULL);
}

NeuralNetwork::NeuralNetwork(OpenCL *context) {
	this->context = context;
	if (!context->isCreated()) {
		cout << "OpenCL not supported. NeuralNetwork object cannot be created properly." << endl;
		return;
	}
	input = new Ptr_List<graph_point*>();
	graph_points =new vector<graph_point*>();
	connections = new vector<connection*>();
	output = NULL;
}
NeuralNetwork::~NeuralNetwork(){
	input->clear(false);
	delete input;
	for (int i = 0; i < graph_points->size(); i++) {
		(*graph_points)[i]->in->clear(false);
		(*graph_points)[i]->out->clear(false);
		delete (*graph_points)[i]->in;
		delete (*graph_points)[i]->out;
		if ((*graph_points)[i]->layer_mem) {
			clReleaseMemObject((*graph_points)[i]->layer_mem);
		}
		free((*graph_points)[i]);
	}
	delete graph_points;
	for (int i = 0; i < connections->size(); i++) {
		free((*connections)[i]->biases.data);
		clReleaseMemObject((*connections)[i]->bias_mem);
		clReleaseMemObject((*connections)[i]->mat_mem);
		free((*connections)[i]->connection_weights.data);
		free((*connections)[i]);
	}
	delete connections;
	if (output_mem) {
		clReleaseMemObject(output_mem);
	}
	if (output_data) {
		free(output_data);
	}
};

void NeuralNetwork::addLayer(uint32_t layer_id, uint32_t layer_size, cl_kernel activation) {
	if (!context->isCreated()) {
		cout << "OpenCL not supported. NeuralNetwork::addLayer cannot be executed" << endl;
		return;
	}
	graph_point* curr=(graph_point*)malloc(sizeof(graph_point*));
	curr->id = layer_id;
	if (insert_graph_point(curr)) {
		curr->in= new Ptr_List<connection*>();
		curr->visited = VISIT_STATE_UNSEEN;
		curr->out = new Ptr_List<connection*>();
	}else {
		cout << "A layer with the id: " << layer_id << " already exists." << endl;
		free(curr);
	}
}

void NeuralNetwork::addInputLayer(uint32_t layer_id, uint32_t layer_size){
	if (!context->isCreated()) {
		cout << "OpenCL not supported. NeuralNetwork::addInputLayer cannot be executed" << endl;
		return;
	}
	graph_point* curr = (graph_point*)malloc(sizeof(graph_point*));
	curr->id = layer_id;
	if (insert_graph_point(curr)) {
		curr->layer_size = layer_size;
		curr->out = new Ptr_List<connection*>();
		curr->visited = VISIT_STATE_FINISHED;
		curr->layer_mem=clCreateBuffer(context->getContext(),CL_MEM_READ_WRITE,sizeof(float)*curr->kernel_layer_size,NULL,NULL);
		input->push_back(curr);
	} else {
		cout << "A layer with the id: " << layer_id << " already exists." << endl;
		free(curr);
	}
}

bool NeuralNetwork::findConnectionById(uint32_t id, uint32_t *loc) {
	int S = 0;
	int L = connections->size() - 1;
	int M = (S + L) / 2;
	while (S <= L) {
		if ((*connections)[M]->id == id) {
			*loc = M;
			return true;
		}
		else if ((*connections)[M]->id > id) {
			L = M;
		}
		else if ((*connections)[M]->id < id) {
			S = M + 1;
		}
		M = (S + L) / 2;
		if (L == M && M == S) {
			*loc = M;
			return (*connections)[M]->id == id;
		}
	}
	*loc = M;
	return false;
}

bool NeuralNetwork::find_connection(connection *index, uint32_t *loc) {
	if (connections->size() == 0) {
		*loc = 0;
		return false;
	}
	int S = 0;
	int L = connections->size() - 1;
	int M = (S + L) / 2;
	while (S <= L) {
		if (*(*connections)[M] == *index) {
			*loc = M;
			return true;
		}
		else if (*(*connections)[M] > *index) {
			L = M;
		}
		else if (*(*connections)[M] < *index) {
			S = M + 1;
		}
		M = (S + L) / 2;
		if (L == M && M == S) {
			*loc = M;
			return *(*connections)[M] == *index;
		}
	}
	*loc = M;
	return false;
}

bool NeuralNetwork::insert_connection(connection *index) {
	uint32_t loc;
	connection connect = *index;
	if (connections->size() == 0) {
		connections->push_back(index);
		return true;
	}
	if (connections->size() == 1) {
		if (*(*connections)[0] > connect) {
			connections->insert(connections->begin(), index);
		}
		else {
			connections->push_back(index);
		}
		return true;
	}
	if (!find_connection(index, &loc)) {
		if (loc == connections->size() - 1) {
			if (*(*connections)[loc] > connect) {
				connections->insert(connections->begin() + loc, index);
			}
			else {
				connections->push_back(index);
			}
			return true;
		}
		if (*(*connections)[loc] > connect) {
			connections->insert(connections->begin() + loc, index);
		}
	else {
			connections->insert(connections->begin() + (loc + 1), index);
		}
		return true;
	}
	else {
		return false;
	}
}

void NeuralNetwork::init() {
	if (!context->isCreated()) {
		cout << "OpenCL not supported. The function NeuralNetwork::init cannot be executed" << endl;
		return;
	}
	if (input->size() == 0 || output == NULL) {
		if (input->size() == 0) {
			cout << "No input layer defined.D Define at least one input layer." << endl;
		}
		if (output == NULL) {
			cout << "No output layer defined. Set an output layer" << endl;
		}
		return;
	}
	mt19937 generator(TIME_MILLIS);
	for (int i = 0; i < connections->size(); i++) {
		float sigma = sqrt(2.0f / ((*connections)[i]->from->layer_size + (*connections)[i]->to->layer_size));
		normal_distribution<float> initializer(0.0f, sigma);
		matrix *curr = &(*connections)[i]->connection_weights;
		fvector *currv = &(*connections)[i]->biases;
		//create a memory for the weight_matrices.
		(*connections)[i]->mat_mem = clCreateBuffer(context->getContext(), CL_MEM_READ_WRITE, (curr->kernel_width*curr->height) * sizeof(float), NULL, NULL);
		(*connections)[i]->bias_mem = clCreateBuffer(context->getContext(), CL_MEM_READ_WRITE, currv->kernel_length*sizeof(float), NULL, NULL);
		for (int y = 0; y < curr->height; y++) {
			for (int x = 0; x < curr->width; x++) {
				curr->data[y*curr->width + x] = initializer(generator);
			}
		}
		for (int i = 0; i < currv->kernel_length; i++) {
			if (i < currv->length) {
				currv->data[i]= initializer(generator);
			}
			else {
				currv->data[i] = 0.0f;
			}
		};
	}
}
void NeuralNetwork::copy_to_input(float **data){
	graph_point **curr = input->iterator();
	int i = 0;
	while (curr != NULL) {
		clEnqueueWriteBuffer(context->getQueue(), (*curr)->layer_mem, false,0, sizeof(float)*(*curr)->layer_size,data[i],0,NULL,NULL);
		curr = input->next();
		i++;
	}
	clFinish(context->getQueue());
}

inline void collect_first_connections(Ptr_List<connection**> *conn,Ptr_List<graph_point*> *input) {
	graph_point** curr = input->iterator();
	while (curr != NULL) {
		connection **conn_curr=(*curr)->out->iterator();
		(*curr)->visited=true;
		while (conn_curr != NULL) {
			conn->push_back(conn_curr);
			conn_curr = (*curr)->out->next();
		}
		curr = input->next();
	}
}
inline void step_forward(Ptr_List<connection**> *conn,OpenCL *context,cl_kernel kernels[2]) {
	connection ***curr = conn->iterator();
	while (curr != NULL) {
		uint8_t visited = (**curr)->to->visited;
		graph_point* target = (**curr)->to;

		cl_kernel kernel=visited==VISIT_STATE_UNSEEN?kernels[0]:kernels[1];
		size_t global[] = { (**curr)->connection_weights.kernel_width};
		size_t local[] = { context->getTileSize() };
		clEnqueueNDRangeKernel(context->getQueue(), kernel, 1, NULL, global, local, 0, NULL, NULL);

		curr = conn->next();
	}
}
void NeuralNetwork::forward_propagation(float * data){
	Ptr_List<connection**> *layers = new Ptr_List<connection**>();
	collect_first_connections(layers,input);
	cl_kernel visited[] = { vec_mat_mul,vec_mat_mul_add };
	step_forward(layers, context,visited);
	delete layers;
}