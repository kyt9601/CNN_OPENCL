__inline void atomic_add_l_f(volatile __local float *addr, float val) {
	union {
		unsigned int u32;
		float        f32;
	} next, expected, current;
	
	current.f32    = *addr;
	do {
		expected.f32 = current.f32;
		next.f32     = expected.f32 + val;
		current.u32  = atomic_cmpxchg( (volatile __local unsigned int *)addr, 
				expected.u32, next.u32);
	} while ( current.u32 != expected.u32 );
}

__inline void atomic_add_g_f(volatile __global float *addr, float val) {
	union {
		unsigned int u32;
		float        f32;
	} next, expected, current;
	
	current.f32    = *addr;
	do {
		expected.f32 = current.f32;
		next.f32     = expected.f32 + val;
		current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
				expected.u32, next.u32);
	} while ( current.u32 != expected.u32 );
}

/*
 * D = channel size
 * N = width and height of an output image
 * Thus, input is (D, N * 2, N * 2) and output is (D, N, N).
 */
__kernel void pooling_layer(__global float *inputs, __global float *outputs, int D, int N) {
	int h = get_global_id(0), i = get_global_id(1), j = get_global_id(2);
	int k, l;
	__global float *input = inputs + h * N * N * 4;
	__global float *output = outputs + h * N * N;
	float max = 0, pixel = 0;

	// remove k=0->2, l=0->2
	max = input[(i * 2 + 0) * 2 * N + j * 2 + 0];
	pixel = input[(i * 2 + 0) * 2 * N + j * 2 + 1];
	max = (max > pixel) ? max : pixel;
	pixel = input[(i * 2 + 1) * 2 * N + j * 2 + 0];
	max = (max > pixel) ? max : pixel;
	pixel = input[(i * 2 + 1) * 2 * N + j * 2 + 1];
	max = (max > pixel) ? max : pixel;
	output[i * N + j] = max;
}

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
#define ReLU(x) (((x)>0)?(x):0)
__kernel void convolution_layer(__global float *inputs, __global float *outputs,
							__constant float *filters, __constant float *biases,
							int D2, int D1, int N) {
	int i, j;
	int k, l, x, y;
	int row, col;
	int counter;
	float sum, t;
	__global float *input, *output;
	__constant float *filter;
	__local float l_input[4608];	// 3 * 3 * 512
	__local float l_sum;
	
	j = get_group_id(0);	// 몇번째 입력 픽셀
	i = get_local_id(0);	// 몇번째 입력 채널
	row = j / N;
	col = j % N;
	input = inputs + j + i * N * N - N - 1;

	x = row - 1; y = col - 1;
	l_input[9 * i] = (0 <= x && x < N && 0 <= y && y < N ? input[0] : 0);
	
	x = row - 1; y = col;
	l_input[9 * i + 1] = (0 <= x && x < N && 0 <= y && y < N ? input[1] : 0);
	
	x = row - 1; y = col + 1;
	l_input[9 * i + 2] = (0 <= x && x < N && 0 <= y && y < N ? input[2] : 0);
	
	x = row; y = col - 1;
	l_input[9 * i + 3] = (0 <= x && x < N && 0 <= y && y < N ? input[N] : 0);
	
	x = row; y = col;
	l_input[9 * i + 4] = (0 <= x && x < N && 0 <= y && y < N ? input[N + 1] : 0);
	
	x = row; y = col + 1;
	l_input[9 * i + 5] = (0 <= x && x < N && 0 <= y && y < N ? input[N + 2] : 0);
	
	x = row + 1; y = col - 1;
	l_input[9 * i + 6] = (0 <= x && x < N && 0 <= y && y < N ? input[2 * N] : 0);
	
	x = row + 1; y = col;
	l_input[9 * i + 7] = (0 <= x && x < N && 0 <= y && y < N ? input[2 * N + 1] : 0);
	
	x = row + 1; y = col + 1;
	l_input[9 * i + 8] = (0 <= x && x < N && 0 <= y && y < N ? input[2 * N + 2] : 0);

	if (256 < D1) {
		input = inputs + j + (i + 256) * N * N - N - 1;

		x = row - 1; y = col - 1;
		l_input[9 * (i + 256)] = (0 <= x && x < N && 0 <= y && y < N ? input[0] : 0);
		
		x = row - 1; y = col;
		l_input[9 * (i + 256) + 1] = (0 <= x && x < N && 0 <= y && y < N ? input[1] : 0);
		
		x = row - 1; y = col + 1;
		l_input[9 * (i + 256) + 2] = (0 <= x && x < N && 0 <= y && y < N ? input[2] : 0);
		
		x = row; y = col - 1;
		l_input[9 * (i + 256) + 3] = (0 <= x && x < N && 0 <= y && y < N ? input[N] : 0);
		
		x = row; y = col;
		l_input[9 * (i + 256) + 4] = (0 <= x && x < N && 0 <= y && y < N ? input[N + 1] : 0);
		
		x = row; y = col + 1;
		l_input[9 * (i + 256) + 5] = (0 <= x && x < N && 0 <= y && y < N ? input[N + 2] : 0);
		
		x = row + 1; y = col - 1;
		l_input[9 * (i + 256) + 6] = (0 <= x && x < N && 0 <= y && y < N ? input[2 * N] : 0);
		
		x = row + 1; y = col;
		l_input[9 * (i + 256) + 7] = (0 <= x && x < N && 0 <= y && y < N ? input[2 * N + 1] : 0);
		
		x = row + 1; y = col + 1;
		l_input[9 * (i + 256) + 8] = (0 <= x && x < N && 0 <= y && y < N ? input[2 * N + 2] : 0);
	}
	
	for (counter = 0; counter < D2; counter++) {
		l_sum = sum = 0;
//		barrier(CLK_LOCAL_MEM_FENCE);
		filter = filters + 9 * (counter * D1 + i);
		
		sum += (l_input[9 * i] == 0 ? 0 : l_input[9 * i] * filter[0]);
		sum += (l_input[9 * i + 1] == 0 ? 0 : l_input[9 * i + 1] * filter[1]);
		sum += (l_input[9 * i + 2] == 0 ? 0 : l_input[9 * i + 2] * filter[2]);
		sum += (l_input[9 * i + 3] == 0 ? 0 : l_input[9 * i + 3] * filter[3]);
		sum += (l_input[9 * i + 4] == 0 ? 0 : l_input[9 * i + 4] * filter[4]);
		sum += (l_input[9 * i + 5] == 0 ? 0 : l_input[9 * i + 5] * filter[5]);
		sum += (l_input[9 * i + 6] == 0 ? 0 : l_input[9 * i + 6] * filter[6]);
		sum += (l_input[9 * i + 7] == 0 ? 0 : l_input[9 * i + 7] * filter[7]);
		sum += (l_input[9 * i + 8] == 0 ? 0 : l_input[9 * i + 8] * filter[8]);

		if (256 < D1) {
			filter = filters + 9 * (counter * D1 + i + 256);

			sum += (l_input[9 * (i + 256)] == 0 ? 0 : l_input[9 * (i + 256)] * filter[0]);
			sum += (l_input[9 * (i + 256) + 1] == 0 ? 0 : l_input[9 * (i + 256) + 1] * filter[1]);
			sum += (l_input[9 * (i + 256) + 2] == 0 ? 0 : l_input[9 * (i + 256) + 2] * filter[2]);
			sum += (l_input[9 * (i + 256) + 3] == 0 ? 0 : l_input[9 * (i + 256) + 3] * filter[3]);
			sum += (l_input[9 * (i + 256) + 4] == 0 ? 0 : l_input[9 * (i + 256) + 4] * filter[4]);
			sum += (l_input[9 * (i + 256) + 5] == 0 ? 0 : l_input[9 * (i + 256) + 5] * filter[5]);
			sum += (l_input[9 * (i + 256) + 6] == 0 ? 0 : l_input[9 * (i + 256) + 6] * filter[6]);
			sum += (l_input[9 * (i + 256) + 7] == 0 ? 0 : l_input[9 * (i + 256) + 7] * filter[7]);
			sum += (l_input[9 * (i + 256) + 8] == 0 ? 0 : l_input[9 * (i + 256) + 8] * filter[8]);
		}
//		atomic_add_l_f(&l_sum, sum);
		l_sum += sum;
//		barrier(CLK_LOCAL_MEM_FENCE);
		t = l_sum + biases[counter];
		output = outputs + N * N * counter;
		output[j] = ReLU(t);
//		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

/*
 * M = output size
 * N = input size
 */
__kernel void fc_layer(__global float *input_neuron, __global float *output_neuron,
							__global float *weights, __global float *biases, int M, int N) {
	int i = get_global_id(1), j = get_global_id(0);
	__local float l_sum;

	if (i == 0)
		l_sum = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	
//	local size = {1, 64}
	if (i < 64)
/*		atomic_add_l_f(&l_sum, input_neuron[i] * weights[j * N + i]
				+ input_neuron[i + 256] * weights[j * N + i + 256]); */
		atomic_add_l_f(&l_sum,
				input_neuron[i] * weights[j * N + i]
				+ input_neuron[i + 64] * weights[j * N + i + 64]
				+ input_neuron[i + 128] * weights[j * N + i + 128]
				+ input_neuron[i + 192] * weights[j * N + i + 192]
				+ input_neuron[i + 256] * weights[j * N + i + 256]
				+ input_neuron[i + 320] * weights[j * N + i + 320]
				+ input_neuron[i + 384] * weights[j * N + i + 384]
				+ input_neuron[i + 448] * weights[j * N + i + 448]);
	barrier(CLK_LOCAL_MEM_FENCE);

	if (i == 0) {
		l_sum += biases[j];
		output_neuron[j] = ( (l_sum>0) ? l_sum : 0 );
	}
}