#include "pch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define CHECK_ERROR(err) \
	if(err != CL_SUCCESS) { \
		printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
		exit(EXIT_FAILURE); \
	}
#define CHECK_BUILD_ERROR(err) \
	if (err == CL_BUILD_PROGRAM_FAILURE) { \
		size_t log_size; \
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size); \
		char *log = (char*)malloc(log_size); \
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL); \
		printf("%s\n", log); \
		free(log); \
	}

inline void softmax(float *output, int N);
inline int find_max(float *fc, int N);

extern const int NETWORK_SIZES[];

cl_uint num_platforms;
cl_platform_id *platforms;
cl_uint num_devices;
cl_device_id *devices;
cl_device_type device_type;
size_t max_work_group_size;
cl_ulong global_mem_size;
cl_ulong local_mem_size;
cl_ulong max_mem_alloc_size;
cl_int err;

cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel convolution_layer, pooling_layer, fc_layer;

char *get_source_code(const char *file_name, size_t *len) {
	char *source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;

	FILE *file = fopen(file_name, "r");

	if (file == NULL) {
		printf("[%s:%d] Failed to open %s ", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);

	for (int i = 0; i < length; i++) {
		buf[0] = source_code[i];
		if (buf[0] == '\n') cnt++;
	}

	source_code[length - cnt] = '\0';
	*len = length - cnt;

	fclose(file);

	return source_code;
}

void cnn_init() {
	size_t kernel_source_size;
	char *kernel_source = get_source_code("kernel.cl", &kernel_source_size), buf[1024];

	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err);

	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_ERROR(err);

	err = clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, 1024, buf, NULL);
	CHECK_ERROR(err);
	printf("CL_PLATFORM_NAME\t: %s\n", buf);

	err = clGetPlatformInfo(platforms[0], CL_PLATFORM_VENDOR, 1024, buf, NULL);
	CHECK_ERROR(err);
	printf("CL_PLATFORM_VENDOR\t: %s\n", buf);

	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	CHECK_ERROR(err);

	devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
	CHECK_ERROR(err);

	err = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 1024, buf, NULL);
	CHECK_ERROR(err);
	printf("CL_DEVICE_NAME\t: %s\n", buf);

	err = clGetDeviceInfo(devices[0], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	CHECK_ERROR(err);
	if (device_type & CL_DEVICE_TYPE_GPU) printf("CL_DEVICE_TYPE\t: GPU\n");
	else { puts("CL_DEVICE_TYPE IS NOT GPU"); exit(-1); }

	err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
	CHECK_ERROR(err);
	printf("CL_DEVICE_MAX_WORK_GROUP_SIZE\t: %lu\n", max_work_group_size);

	err = clGetDeviceInfo(devices[0], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
	CHECK_ERROR(err);
	printf("CL_DEVICE_GLOBAL_MEM_SIZE\t: %lu\n", global_mem_size);

	err = clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
	CHECK_ERROR(err);
	printf("CL_DEVICE_LOCAL_MEM_SIZE\t: %lu\n", local_mem_size);

	err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, NULL);
	CHECK_ERROR(err);
	printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE\t: %lu\n\n", local_mem_size);

	context = clCreateContext(NULL, 1, devices, NULL, NULL, &err);
	CHECK_ERROR(err);

	queue = clCreateCommandQueueWithProperties(context, *devices, 0, &err);
	CHECK_ERROR(err);

	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(program, 1, devices, "-cl-no-signed-zeros -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only", NULL, NULL);
	CHECK_BUILD_ERROR(err);
	CHECK_ERROR(err);

	convolution_layer = clCreateKernel(program, "convolution_layer", &err);
	CHECK_ERROR(err);
	pooling_layer = clCreateKernel(program, "pooling_layer", &err);
	CHECK_ERROR(err);
	fc_layer = clCreateKernel(program, "fc_layer", &err);
	CHECK_ERROR(err);
}

void cnn(float *images, float **network, int *labels, float *confidences, int num_images) {
	int i, j, h, cnt;
	int num_of_conv_layers[] = { 3, 2, 2, 3, 3, 3 };
	int size_of_conv_layers[6] = {
		0 * 0 * 0,
		64 * 32 * 32,
		128 * 16 * 16,
		256 * 8 * 8,
		512 * 4 * 4,
		512 * 2 * 2
	};
	int conv_layer_args[6][3] = {
		{ 0, 0, 0 },
		{ 64, 3, 32 },
		{ 128, 64, 16 },
		{ 256, 128, 8 },
		{ 512, 256, 4 },
		{ 512, 512, 2 }
	};
	int size_of_pooling_layers[6][3] = {
		{ 3, 32, 32 },
		{ 64, 16, 16 },
		{ 128, 8, 8 },
		{ 256, 4, 4 },
		{ 512, 2, 2 },
		{ 512, 1, 1 }
	};
	int size_of_fc_layers[4] = {
		0,
		512,
		512,
		10
	};

	float fc3[10];

	size_t global_size, global_size2[2], global_size3[3];
	size_t local_size, local_size2[2], local_size3[3];
	cl_mem W[6][4], B[6][4];
	cl_mem C[6][4];
	cl_mem P[6];
	cl_mem FC[4];

	cnt = 0;
	for (i = 1; i <= 6; i++)
		for (j = 1; j <= num_of_conv_layers[i % 6]; j++) {
			W[i % 6][j] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * NETWORK_SIZES[cnt], network[cnt++], &err);
			CHECK_ERROR(err);
			B[i % 6][j] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * NETWORK_SIZES[cnt], network[cnt++], &err);
			CHECK_ERROR(err);
		}

	for (i = 1; i < 6; i++)
		for (j = 1; j <= num_of_conv_layers[i]; j++) {
			C[i][j] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size_of_conv_layers[i], NULL, &err);
			CHECK_ERROR(err);
		}

	for (i = 0; i < 6; i++) {
		P[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size_of_pooling_layers[i][0] *
				size_of_pooling_layers[i][1] * size_of_pooling_layers[i][2], NULL, &err);
		CHECK_ERROR(err);
	}

	for (i = 1; i < 4; i++) {
		FC[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size_of_fc_layers[i], NULL, &err);
		CHECK_ERROR(err);
	}

	for (i = 0; i < num_images; ++i) {
		float *image = images + i * 3 * 32 * 32;
		err = clEnqueueWriteBuffer(queue, P[0], CL_TRUE, 0, sizeof(float) * size_of_pooling_layers[0][0] *
				size_of_pooling_layers[0][1] * size_of_pooling_layers[0][2], image, 0, NULL, NULL);
		CHECK_ERROR(err);

		for (j = 1; j <= 5; j++) {
			for (h = 1; h <= num_of_conv_layers[j]; h++) {
				if (h == 1) {
//					convolution_layer(p[j - 1], c[j][h], w[j][h], b[j][h], 64, 3, 32);
					err = clSetKernelArg(convolution_layer, 0, sizeof(cl_mem), P + (j - 1));					CHECK_ERROR(err);
					err = clSetKernelArg(convolution_layer, 5, sizeof(int), &conv_layer_args[j][1]);					CHECK_ERROR(err);
					global_size = (conv_layer_args[j][1] < 512 ? conv_layer_args[j][1] : 256) * conv_layer_args[j][2] * conv_layer_args[j][2];
					local_size = (conv_layer_args[j][1] < 512 ? conv_layer_args[j][1] : 256);
				}
				else {
//					convolution_layer(c[j][h - 1], c[j][h], w[j][h], b[j][h], 64, 3, 32);
					err = clSetKernelArg(convolution_layer, 0, sizeof(cl_mem), &C[j][h-1]);					CHECK_ERROR(err);
					err = clSetKernelArg(convolution_layer, 5, sizeof(int), &conv_layer_args[j][0]);					CHECK_ERROR(err);
					global_size = (conv_layer_args[j][0] < 512 ? conv_layer_args[j][0] : 256) * conv_layer_args[j][2] * conv_layer_args[j][2];
					local_size = (conv_layer_args[j][0] < 512 ? conv_layer_args[j][0] : 256);
				}

				err = clSetKernelArg(convolution_layer, 1, sizeof(cl_mem), &C[j][h]);				CHECK_ERROR(err);
				err = clSetKernelArg(convolution_layer, 2, sizeof(cl_mem), &W[j][h]);				CHECK_ERROR(err);
				err = clSetKernelArg(convolution_layer, 3, sizeof(cl_mem), &B[j][h]);				CHECK_ERROR(err);
				err = clSetKernelArg(convolution_layer, 4, sizeof(int), &conv_layer_args[j][0]);				CHECK_ERROR(err);
				err = clSetKernelArg(convolution_layer, 6, sizeof(int), &conv_layer_args[j][2]);				CHECK_ERROR(err);
//				time_t start = clock();
				clEnqueueNDRangeKernel(queue, convolution_layer, 1, NULL, &global_size, &local_size, 0, NULL, NULL);				CHECK_ERROR(err);
				err = clFinish(queue);				CHECK_ERROR(err);
//				printf("image%d [%d][%d] time : %lf\n", i, j, h, (clock() - start) / (double)CLOCKS_PER_SEC);
			}

			err = clSetKernelArg(pooling_layer, 0, sizeof(cl_mem), &C[j][h-1]);			CHECK_ERROR(err);
			err = clSetKernelArg(pooling_layer, 1, sizeof(cl_mem), P + j);			CHECK_ERROR(err);
			err = clSetKernelArg(pooling_layer, 2, sizeof(int), &size_of_pooling_layers[j][0]);			CHECK_ERROR(err);
			err = clSetKernelArg(pooling_layer, 3, sizeof(int), &size_of_pooling_layers[j][1]);			CHECK_ERROR(err);
			global_size3[0] = size_of_pooling_layers[j][0]; global_size3[1] = size_of_pooling_layers[j][1]; global_size3[2] = size_of_pooling_layers[j][2];
			local_size3[0] = 4; local_size3[1] = 4; local_size3[2] = 4;
			clEnqueueNDRangeKernel(queue, pooling_layer, 3, NULL, global_size3, local_size3, 0, NULL, NULL);			CHECK_ERROR(err);
			err = clFinish(queue);			CHECK_ERROR(err);
		}

		// 3 번의 fully-connected layer
		for (h = 1; h <= 3; h++) {
			if (h == 1) {
//				fc_layer(p[5], fc[h], w[0][h], b[0][h], size_of_fc_layers[h], 512);
				err = clSetKernelArg(fc_layer, 0, sizeof(cl_mem), P + 5);				CHECK_ERROR(err);
			}
			else {
//				fc_layer(fc[h-1], fc[h], w[0][h], b[0][h], size_of_fc_layers[h], 512);
				err = clSetKernelArg(fc_layer, 0, sizeof(cl_mem), FC + (h - 1));				CHECK_ERROR(err);
			}

			err = clSetKernelArg(fc_layer, 1, sizeof(cl_mem), FC + h);			CHECK_ERROR(err);
			err = clSetKernelArg(fc_layer, 2, sizeof(cl_mem), &W[0][h]);			CHECK_ERROR(err);
			err = clSetKernelArg(fc_layer, 3, sizeof(cl_mem), &B[0][h]);			CHECK_ERROR(err);
			err = clSetKernelArg(fc_layer, 4, sizeof(int), size_of_fc_layers + h);			CHECK_ERROR(err);
			int N = 512;
			err = clSetKernelArg(fc_layer, 5, sizeof(int), &N);			CHECK_ERROR(err);
			global_size2[0] = size_of_fc_layers[h]; global_size2[1] = N;
			local_size2[0] = 1; local_size2[1] = 64;
			clEnqueueNDRangeKernel(queue, fc_layer, 2, NULL, global_size2, local_size2, 0, NULL, NULL);			CHECK_ERROR(err);
			err = clFinish(queue);			CHECK_ERROR(err);
		}

		err = clEnqueueReadBuffer(queue, FC[3], CL_TRUE, 0, sizeof(float) * size_of_fc_layers[3], fc3, 0, NULL, NULL);		CHECK_ERROR(err);
		softmax(fc3, 10);
		labels[i] = find_max(fc3, 10);
		confidences[i] = fc3[labels[i]];
	}

	for (i = 0; i < 6; i++)
		for (j = 1; j <= num_of_conv_layers[i]; j++) {
			err = clReleaseMemObject(W[i][j]);	CHECK_ERROR(err);
			err = clReleaseMemObject(B[i][j]);	CHECK_ERROR(err);
			if (0 < i) { err = clReleaseMemObject(C[i][j]);	CHECK_ERROR(err); }
			if (i == 0) { err = clReleaseMemObject(FC[j]);	CHECK_ERROR(err); } // i == 0 일때 1~3 까지 총 3번 실행됨
		}
		err = clReleaseMemObject(P[i]);	CHECK_ERROR(err);

	err = clReleaseKernel(convolution_layer);	CHECK_ERROR(err);
	err = clReleaseKernel(pooling_layer);	CHECK_ERROR(err);
	err = clReleaseKernel(fc_layer);	CHECK_ERROR(err);
	err = clReleaseProgram(program);	CHECK_ERROR(err);
	err = clReleaseCommandQueue(queue);	CHECK_ERROR(err);
	err = clReleaseContext(context);	CHECK_ERROR(err);

	free(devices);
	free(platforms);
}

inline void softmax(float *output, int N) {
	int i;
	float max = output[0];
	float sum = 0;

	for (i = 1; i < N; i++)
		max = (output[i] > max) ? output[i] : max;

	for (i = 0; i < N; i++)
		sum += exp(output[i] - max);

	for (i = 0; i < N; i++)
		output[i] = exp(output[i] - max) / sum;
}

inline int find_max(float *fc, int N) {
	int i;
	int maxid = 0;
	float maxval = 0;

	for (i = 0; i < N; i++)
		if (maxval < fc[i]) {
			maxval = fc[i];
			maxid = i;
		}

	return maxid;
}