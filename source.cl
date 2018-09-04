/*
#define PLATFORM_NAME \
for (int i = 0; i < platform_c; i++) {\
	uint32_t size;\
	clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &size);\
	void* name = malloc(size);\
	clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, size, name, &size);\
	cout << (char*)name << endl;\
}

#define DEVICE_NAME \
for (int i = 0; i < gpu_c; i++) {\
	uint32_t size;\
	clGetDeviceInfo(gpus[i], CL_DEVICE_NAME, 0, NULL, &size);\
	void* name = malloc(size);\
	clGetDeviceInfo(gpus[i], CL_DEVICE_NAME, size, name, &size);\
	\
	clGetDeviceInfo(gpus[i], CL_DEVICE_VENDOR, 0, NULL, &size);\
	void* vendor = malloc(size);\
	clGetDeviceInfo(gpus[i], CL_DEVICE_VENDOR, size, vendor, &size);\
	cout <<(char*)vendor<<" "<<(char*)name << endl;\
	free(name);\
	free(vendor);\
}
*/
__kernel void vec_mat_mul(const int L,
							const int M,
							__global float* layer,
							__global float *matrix,
							__global float *result){
	const int localV=get_local_id(0);
	const int matRow=TS*get_group_id(0)+localV;
	const int matCol=get_global_id(1);
	
	__local float inVecD[TS];
	__local float matColD[TS];
	
	float vec[WPT];
	float mat[WPT];
	
	matColD[localV]=matrix[M*matRow+matCol];
	inVecD[localV]=layer[matRow];
	
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int i=0;i<TS/WPT;i++){
		for(int j=0;j<WPT;j++){
			vec[j]=inVecD[i*WPT+j];
			mat[j]=matColD[i*WPT+j];
		}
		for(int j=0;j<WPT;j++){
			result[matCol]+=vec[j]*mat[j];
		}
	}
}
/*
M: The number of rows in the first matrix, and in the result matrix.
N: The number of columns in the second matrix,a nd in the result matrix.
K: The number of rows in the second matrx, and the number of columns in the first matrix.
*/
__kernel void padding(const int M,const int N,
					const int P,const int Q,
					__global float *input,
					__global float* output){
	const int tmInd=get_local_id(0);
	const int tnInd=get_local_id(1);
	const int grIndM=get_group_id(0);
	const int grIndN=get_group_id(1);
	
	int x=tmInd+TSR*grIndM;
	int y=tnInd+TSR*grIndN;
	if(x<M&&y<N){
		output[P*y+x]=input[M*y+x];
	}else{
		output[P*y+x]=0.0f;
	}
}

__kernel void mat_mult(const int M,const int N,const int K,
					global float *first,
					global float *second,
					global float *result){
	const int mtId=get_local_id(0);
	const int ntId=get_local_id(1);
	const int grR=get_group_id(0);
	const int grI=get_group_id(1);
	
	__local float localA[TSR][TSI];
	__local float localB[TSI][TSR];
	
	float Areg;
	float Breg[WPT];
	float acc[WPT][WPT];
	
	for(int y=0;y<WPT;y++){
		for(int x=0;x<WPT;x++){
			acc[x][y]=0.0f;
		}
	}
	int tile_size=TSR*TSI;
	for(int i=0;i<K/TSR;i++){
		for(int j=0;j<tile_size;j++){
			int rowA=tile_size%TSR;
			int colA=tile_size/TSR;
			localA[colA][rowA]=first[K*(TSR*grM+tind)]
		}
		
		barrier(CLK_LOCAL_MEM_FENCE)
	}
	//float acc[WPT];
	result[mtId]=first[mtId]+second[mtId];
}