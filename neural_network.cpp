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

void NeuralNetwork::softmax() {
	size_t size[] = { output->kernel_layer_size };
	size_t sizel[] = { context->getTileSize() };

	//raise the output to the power of E (Euler-number)
	powOfE(context,softmax_pow,output_mem,output,size,output_data);
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
	output_data = (float*)malloc(sizeof(float)*layer_size);
	output = (graph_point*)malloc(sizeof(graph_point));
	output->id = layer_id;
	if (insert_graph_point(output)) {
		output->layer_size=layer_size;
		if (layer_size % 32 != 0) {
			layer_size += (32 - layer_size % 32);
		}
		output_mem = clCreateBuffer(context->getContext(), CL_MEM_READ_WRITE, layer_size * sizeof(float), NULL, NULL);
		output->kernel_layer_size = layer_size;
		output->in = new Ptr_List<connection*>();
		output->visited = false;
	}else {
		cout << "A layer with the id: " << layer_id << " already exists." << endl;
		free(output);
	};
	for (int i = 0; i < layer_size; i++) {
		if (i < output->layer_size) {
			output_data[i] = i;
		}else {
			output_data[i] = 0.0f;
		}
		cout << output_data[i] << " ";
	}
	cout << endl;
	softmax();
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

	reduce_sum = clCreateKernel(context->getProgram(), "reduce_sum", NULL);
	softmax_pow = clCreateKernel(context->getProgram(), "softmax_pow", NULL);
	skalar_div = clCreateKernel(context->getProgram(), "skalar_div", NULL);
	vec_mat_mul = clCreateKernel(context->getProgram(), "vec_mat_mul", NULL);
};

void NeuralNetwork::addLayer(uint32_t layer_id, uint32_t layer_size, cl_kernel activation) {
	graph_point* curr=(graph_point*)malloc(sizeof(graph_point*));
	curr->id = layer_id;
	if (insert_graph_point(curr)) {
		curr->in= new Ptr_List<connection*>();
		curr->visited = false;
		curr->out = new Ptr_List<connection*>();
	}else {
		cout << "A layer with the id: " << layer_id << " already exists." << endl;
		free(curr);
	}
}

void NeuralNetwork::addInputLayer(uint32_t layer_id, uint32_t layer_size){
	graph_point* curr = (graph_point*)malloc(sizeof(graph_point*));
	curr->id = layer_id;
	if (insert_graph_point(curr)) {
		curr->layer_size = layer_size;
		curr->visited = false;
		curr->out = new Ptr_List<connection*>();
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

void NeuralNetwork::init(){
	if (!context->isCreated()) {
		cout << "OpenCL not supported. The function NeuralNetwork::init cannot be executed" << endl;
		return;
	}
	if (input->size() == 0 || output == NULL) {
		if (input->size() == 0) {
			cout << "No input layer defined.D Define at least one input layer." << endl;
		}
		if (output == NULL) {
			cout << " No output layer defined. Set an output layer" << endl;
		}
		return;
	}
	mt19937 generator(TIME_MILLIS);
	for (int i = 0; i < connections->size(); i++) {
		float sigma = sqrt(2.0f/((*connections)[i]->from->layer_size + (*connections)[i]->to->layer_size));
		normal_distribution<float> initializer(0.0f, sigma);
		matrix *curr= &(*connections)[i]->connection_weights;
		(*connections)[i]->memory = clCreateBuffer(context->getContext(), CL_MEM_READ_WRITE, curr->kernel_width*curr->height*sizeof(float),NULL,NULL);
		for (int y = 0; y < curr->height; y++) {
			for (int x = 0; x < curr->width; x++) {
				curr->data[y*curr->width+x]=initializer(generator);
			}
		}
	}
	uint32_t max_layer_size=(*graph_points)[0]->layer_size;
	for (int i = 1; i < graph_points->size(); i++) {
		if (max_layer_size < (*graph_points)[i]->kernel_layer_size) {
			max_layer_size = (*graph_points)[i]->kernel_layer_size;
		}
	}
	/*layer_data = (float*)malloc(sizeof(float)*max_layer_size);
	for (int i = 0; i < max_layer_size; i++) {
		layer_data[i] = 0.0f;
	}
	data_mem = clCreateBuffer(context->getContext(), CL_MEM_READ_WRITE, max_layer_size * sizeof(float), NULL, NULL);*/
}
/*
clEnqueueWriteBuffer(context->getQueue(), (*(*curr)->out)[0]->memory, false, 0, sizeof(float)*(*curr)->layer_size, NULL, 0, NULL, NULL);
clSetKernelArg(vec_mat_mul, 0, );
*/
void NeuralNetwork::forward_propagation(float * data){
	Ptr_List<connection**> *searching = new Ptr_List<connection**>();
	graph_point **curr = input->iterator();
	while (curr != NULL) {
		connection **item=(*curr)->out->iterator();
		while (item != NULL) {
			searching->push_back(item);
			item= (*curr)->out->next();
		}
		curr = input->next();
	}
	connection ***curr_c=searching->iterator();
	while (curr_c!=NULL) {
		
	}
	delete searching;
}

uint32_t graph_point::max_outgoing_count(){
	Ptr_List<connection**> *conns = new Ptr_List<connection**>();
	return uint32_t();
}
