#define E 2.718281828f

__kernel void vec_mat_mul(	const int L,
							const int M,
							__global float *layer,
							__global float *matrix,
							__global float *result){
	int m=get_global_id(0);
	int lm=get_local_id(0);
	
	__local float locals[TS];
	
	float vReg=0.0f;
	float mReg[WPT];
	float acc[WPT];
	
	for(int i=0;i<L;i++){
		locals[lm]=matrix[i*M+m];
		vReg=layer[i];
		for(int j=0;j<WPT;j++){
			mReg[j]=locals[lm];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for(int j=0;j<WPT;j++){
			acc[j]+=vReg*mReg[j];
		}
	}
	result[m]=acc[m%WPT];
}

__kernel void vec_mat_mul_add(	const int L,
							const int M,
							__global float *layer,
							__global float *matrix,
							__global float *result){
	int m=get_global_id(0);
	int lm=get_local_id(0);
	
	__local float locals[TS];
	
	float vReg=0.0f;
	float mReg[WPT];
	float acc[WPT];
	
	for(int i=0;i<L;i++){
		locals[lm]=matrix[i*M+m];
		vReg=layer[i];
		for(int j=0;j<WPT;j++){
			mReg[j]=locals[lm];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for(int j=0;j<WPT;j++){
			acc[j]+=vReg*mReg[j];
		}
	}
	result[m]+=acc[m%WPT];
}

__kernel void softmax_pow(const int N,	
							__global float *input,
							__global float *output){
	int n=get_global_id(0);
	if(n<N){
		output[n]=pow(E,input[n]);
	}else{
		output[n]=0.0f;
	}
}

__kernel void reduce_sum(const int N,
							__global float *input,
							__global float *output){
	int n=get_global_id(0);
	int localn=get_local_id(0);
	
	__local float tile[TS];
	float sumnreg[TS/WPT][WPT];
	
	tile[localn]=input[n];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int rn=localn/2;
	if(n%2==0){
		output[n/2]=tile[rn*2]+tile[rn*2+1];
		output[N/2+n/2]=0.0f;
	}
}

__kernel void add(__global float *in1,__global float *in2,__global float *out){
	int n=get_global_id(0);
	out[n]=in1[n]+in2[n];
}

__kernel void sub(__global float *in1,__global float *in2,__global float *out){
	int n=get_global_id(0);
	out[n]=in1[n]-in2[n];
}

__kernel void elements_mul(__global float *in1,__global float *in2,__global float *out){
	int n=get_global_id(0);
	out[n]=in1[n]*in2[n];
}

__kernel void one_minus(__global float *in1,__global float *in2,__global float *out){
	int n=get_global_id(0);
	out[n]=1.0f-in2[n];
}

__kernel void skalar_div(const float skalar,__global float *in, __global float *out){
	int n=get_global_id(0);
	out[n]=in[n]/skalar;
}

__kernel void skalar_mul(const float skalar,__global float *in, __global float *out){
	int n=get_global_id(0);
	out[n]=skalar*in[n];
}

__kernel void sigmoidv(__global float* in,__global float *out){
	int n=get_global_id(0);
	out[n]=1.0f/(1.0f-pow(E,-in[n]));
}

__kernel void tanhv(__global float* in,__global float *out){
	int n=get_global_id(0);
	out[n]=tanh(in[n]);
}

__kernel void relu(__global float* in,__global float *out){
	int n=get_global_id(0);
	if(in[n]<0.0f){
		out[n]=0.0f;
	}else{
		out[n]=in[n];
	}
}

__kernel void tanh_deriv(__global float *in,__global float *out){
	int n=get_global_id(0);
	float cosh_val=cosh(in[n]);
	out[n]=1/cosh_val*cosh_val;
}

__kernel void sigmoid_deriv(__global float *in,__global float *out){
	int n=get_global_id(0);
	out[n]=pow(E,-in[n])/pow((pow(E,-in[n])+1),2);
}