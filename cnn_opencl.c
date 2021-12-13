#define _CRT_SECURE_NO_WARNINGS 


#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include "cnn.h"
#include<time.h>
#include<math.h>


extern const int INPUT_DIM[21];
extern const int OUTPUT_DIM[21];
extern const int NBYN[21];

extern const char* CLASS_NAME[];



struct openCLInfo {
    cl_device_id device;
    cl_context context;
    cl_command_queue taskQueue;
    cl_command_queue memoryQueue;
    cl_kernel kernel_conv;
    cl_kernel kernel_maxpool;
    cl_kernel kernel_fc;
};


#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

char* get_source_code(const char* file_name, size_t* len) {
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t length = (size_t)ftell(file);
    rewind(file);

    char* source_code = (char*)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';
    fclose(file);
    *len = length;

    return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char* log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    };
}




static void softmax(float* input, int N) {
    int i;
    float max = input[0];
    for (i = 1; i < N; i++) {
        if (max < input[i]) max = input[i];
    }
    float sum = 0;
    for (i = 0; i < N; i++) {
        sum += exp(input[i] - max);
    }
    for (i = 0; i < N; i++) {
        input[i] = exp(input[i] - max) / (sum + 1e-7);
    }
}

static int find_max(float* input, int classNum) {
    int i;
    int maxIndex = 0;
    float max = 0;
    for (i = 0; i < classNum; i++) {
        if (max < input[i]) {
            max = input[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}


struct openCLInfo openCLInit(const int PLAT_NO, const int DEV_NO) {


    struct openCLInfo cl;

    cl_int err;

    cl_uint num_platforms;
    cl_platform_id* platforms;

    cl_uint num_devices;
    cl_device_id* devices;

    cl_context context;

    cl_command_queue queue;

    cl_program program;


    cl_kernel kernel_integral, kernel_reduction;

    cl_mem bufOutput;

    cl_command_queue taskQueue;
    cl_command_queue memoryQueue;

    char* kernel_source;
    size_t kernel_source_size;



    err = clGetPlatformIDs(0, NULL, &num_platforms);
    CHECK_ERROR(err);

    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    CHECK_ERROR(err);


    //platform check
    size_t plat_name_size;
    clGetPlatformInfo(platforms[PLAT_NO], CL_PLATFORM_NAME, 0, NULL, &plat_name_size);

    char* plat_name = (char*)malloc(plat_name_size);
    clGetPlatformInfo(platforms[PLAT_NO], CL_PLATFORM_NAME, plat_name_size, plat_name, NULL);
    printf("normal_opencl Platform: %s\n", plat_name);


    //device check
    err = clGetDeviceIDs(platforms[PLAT_NO], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    CHECK_ERROR(err);
    devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
    err = clGetDeviceIDs(platforms[PLAT_NO], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    CHECK_ERROR(err);


    size_t dev_name_size;
    clGetDeviceInfo(devices[DEV_NO], CL_DEVICE_NAME, 0, NULL, &dev_name_size);
    char* dev_name = (char*)malloc(dev_name_size);
    clGetDeviceInfo(devices[DEV_NO], CL_DEVICE_NAME, dev_name_size, dev_name, NULL);
    printf("Device: %s\n\n", dev_name);


    context = clCreateContext(NULL, 1, &devices[DEV_NO], NULL, NULL, &err);
    CHECK_ERROR(err);



    taskQueue = clCreateCommandQueue(context, devices[DEV_NO], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    CHECK_ERROR(err);

    memoryQueue = clCreateCommandQueue(context, devices[DEV_NO], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    CHECK_ERROR(err);

    kernel_source = get_source_code("cnn.cl", &kernel_source_size);
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);


    err = clBuildProgram(program, 1, &devices[DEV_NO], "-cl-fast-relaxed-math", NULL, NULL);
    build_error(program, devices[DEV_NO], err);

    cl_kernel kernel_conv;

    kernel_conv = clCreateKernel(program, "convolution", &err);
    CHECK_ERROR(err);

    cl_kernel kernel_maxpool;

    kernel_maxpool = clCreateKernel(program, "max_pooling", &err);
    CHECK_ERROR(err);

    cl_kernel kernel_fc;

    kernel_fc = clCreateKernel(program, "fully_connected", &err);
    CHECK_ERROR(err);



    cl.device = devices[DEV_NO];
    cl.context = context;
    cl.taskQueue = taskQueue;
    cl.memoryQueue = memoryQueue;
    cl.kernel_conv = kernel_conv;
    cl.kernel_maxpool = kernel_maxpool;
    cl.kernel_fc = kernel_fc;

    return cl;


}




void cnn_init(float** network, float** w, float** b, cl_mem* layerBuf, struct openCLInfo cl) {

    cl_int err;
    //TODO
    int offset = 0;
    // link weights and biases to network
    for (int i = 0; i < 17; ++i) {
        if (i == 2 || i == 5 || i == 9 || i == 13) i++;	// pooling layer has no weights and biases
        w[i] = network + offset;
        offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i]; // 3*3짜리 필터가 input dim의 깊이만큼 총 3*3*inputDim의 필터가 outputDim만큼 있음 -> 그리고 이러한 과정을 21번 반복계산함
        b[i] = network + offset; //network의 마지막부분은 항상 가중치가 있음
        offset += OUTPUT_DIM[i]; //그리고 그 가중치는 outputDim의 개수만큼 있음


    }


    for (int i = 18; i < 21; ++i) { //fully connected layer의 weight와 bias를 나타냄
        w[i] = network + offset; //
        offset += INPUT_DIM[i] * OUTPUT_DIM[i];
        b[i] = network + offset;
        offset += OUTPUT_DIM[i];
    }



    for (int i = 0; i < 2; i++) {
        layerBuf[i] = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, sizeof(float) * OUTPUT_DIM[0] * NBYN[0] * NBYN[0], NULL, &err);
        CHECK_ERROR(err);
    }

    CHECK_ERROR(err);


}


void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {


    struct openCLInfo clInfo;

    float* w[21];
    float* b[21];
    float result[10];

    cl_mem layerBuf[2];
    cl_mem wBuf, bBuf, wBuf2, bBuf2;
    cl_int err;
    
    cl_event execEvent[21];
    cl_event memEvent[21][2];


    clInfo = openCLInit(0, 0); // --------------- 해당 플랫폼번호와 디바이스 번호를 통해 플랫폼 아이디와 디바이스 아이디 획득
    cnn_init(network, w, b, layerBuf, clInfo);

    // filter용 buffer와 bias용 버퍼 생성
    wBuf = clCreateBuffer(clInfo.context, CL_MEM_READ_ONLY, sizeof(float) * 3 * 3 * INPUT_DIM[11] * OUTPUT_DIM[11], NULL, &err);
    CHECK_ERROR(err);
    bBuf = clCreateBuffer(clInfo.context, CL_MEM_READ_ONLY, sizeof(float) * OUTPUT_DIM[11], NULL, &err);
    CHECK_ERROR(err);

    wBuf2 = clCreateBuffer(clInfo.context, CL_MEM_READ_ONLY, sizeof(float) * 3 * 3 * INPUT_DIM[11] * OUTPUT_DIM[11], NULL, &err);
    CHECK_ERROR(err);
    bBuf2 = clCreateBuffer(clInfo.context, CL_MEM_READ_ONLY, sizeof(float) * OUTPUT_DIM[11], NULL, &err);
    CHECK_ERROR(err);

    time_t start, end;
    start = clock();

    for (int i = 0; i < num_images; i++) {






        //시작 사진을 버퍼에 올리기
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, layerBuf[0], CL_TRUE, 0, sizeof(float) * INPUT_DIM[0] * NBYN[0] * NBYN[0], images, 0, NULL, NULL);
        CHECK_ERROR(err);



        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[0] * OUTPUT_DIM[0], w[0], 0, NULL, &memEvent[0][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[0], b[0], 0, NULL, &memEvent[0][1]);
        CHECK_ERROR(err);

        cl_int workDevide = 1;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[0] * NBYN[0], NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3, NULL);
        CHECK_ERROR(err);
        // ===========> global[outDim][nbyn][nbyn][workDeivde]  =================> inDim으로부터 나오는 work size를 workDevide만큼 나눠보자.
        // ===========> local[1024/(nbyn*nbyn)/workDevide][nbyn][nbyn][workDevide]

        size_t global_size[3] = { OUTPUT_DIM[0], NBYN[0] * NBYN[0], 1 };
        size_t local_size[3] = { 1024 / (NBYN[0] * NBYN[0]) / workDevide, NBYN[0] * NBYN[0] ,1 };
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[0], &execEvent[0]);
        CHECK_ERROR(err);


        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf2, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[1] * OUTPUT_DIM[1], w[1], 0, NULL, &memEvent[1][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf2, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[1], b[1], 0, NULL, &memEvent[1][1]);
        CHECK_ERROR(err);

        workDevide = 1;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf2);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf2);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[1] * NBYN[1], NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3, NULL);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[1];
        global_size[1] = NBYN[1] * NBYN[1];
        global_size[2] = workDevide;
        local_size[0] = 1024 / (NBYN[1] * NBYN[1]) / workDevide;
        local_size[1] = NBYN[1] * NBYN[1];
        local_size[2] = workDevide;
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[1], &execEvent[1]);
        CHECK_ERROR(err);




        err = clSetKernelArg(clInfo.kernel_maxpool, 0, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_maxpool, 1, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_maxpool, 2, sizeof(cl_int), &NBYN[2]);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[2];
        global_size[1] = NBYN[2] * NBYN[2];
        local_size[0] = 1024 / (NBYN[2] * NBYN[2]);
        local_size[1] = NBYN[2] * NBYN[2];
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_maxpool, 2, NULL,
            global_size, local_size,
            1, &execEvent[1], &execEvent[2]);
        CHECK_ERROR(err);






        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[3] * OUTPUT_DIM[3], w[3], 1, &execEvent[0], &memEvent[3][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[3], b[3], 1, &execEvent[0], &memEvent[3][1]);
        CHECK_ERROR(err);

        workDevide = 2;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[3]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[3]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[3]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[3] * NBYN[3] * workDevide, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3 * 1024 / (NBYN[3] * NBYN[3]) * workDevide, NULL);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[3];
        global_size[1] = NBYN[3] * NBYN[3];
        global_size[2] = workDevide;
        local_size[0] = 1024 / (NBYN[3] * NBYN[3]) / workDevide;
        local_size[1] = NBYN[3] * NBYN[3];
        local_size[2] = workDevide;
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[3], &execEvent[3]);
        CHECK_ERROR(err);






        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf2, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[4] * OUTPUT_DIM[4], w[4], 1, &execEvent[1], &memEvent[4][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf2, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[4], b[4], 1, &execEvent[1], &memEvent[4][1]);
        CHECK_ERROR(err);

        workDevide = 2;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf2);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf2);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[4]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[4]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[4]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[4] * NBYN[4] * workDevide, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3 * 1024 / (NBYN[4] * NBYN[4]) * workDevide, NULL);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[4];
        global_size[1] = NBYN[4] * NBYN[4];
        global_size[2] = workDevide;
        local_size[0] = 1024 / (NBYN[4] * NBYN[4]) / workDevide;
        local_size[1] = NBYN[4] * NBYN[4];
        local_size[2] = workDevide;
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[4], &execEvent[4]);
        CHECK_ERROR(err);





        err = clSetKernelArg(clInfo.kernel_maxpool, 0, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_maxpool, 1, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_maxpool, 2, sizeof(cl_int), &NBYN[5]);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[5];
        global_size[1] = NBYN[5] * NBYN[5];
        local_size[0] = 1024 / (NBYN[5] * NBYN[5]);
        local_size[1] = NBYN[5] * NBYN[5];
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_maxpool, 2, NULL,
            global_size, local_size,
            1, &execEvent[4], &execEvent[5]);
        CHECK_ERROR(err);






        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[6] * OUTPUT_DIM[6], w[6], 1, &execEvent[3], &memEvent[6][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[6], b[6], 1, &execEvent[3], &memEvent[6][1]);
        CHECK_ERROR(err);

        workDevide = 8;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[6]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[6]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[6]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[6] * NBYN[6] * workDevide, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3 * 1024 / (NBYN[6] * NBYN[6]) * workDevide, NULL);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[6];
        global_size[1] = NBYN[6] * NBYN[6];
        global_size[2] = workDevide;
        local_size[0] = 1024 / (NBYN[6] * NBYN[6]) / workDevide;
        local_size[1] = NBYN[6] * NBYN[6];
        local_size[2] = workDevide;
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[6], &execEvent[6]);
        CHECK_ERROR(err);




        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf2, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[7] * OUTPUT_DIM[7], w[7], 1, &execEvent[4], &memEvent[7][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf2, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[7], b[7], 1, &execEvent[4], &memEvent[7][1]);
        CHECK_ERROR(err);

        workDevide = 8;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf2);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf2);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[7]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[7]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[7]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[7] * NBYN[7] * workDevide, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3 * 1024 / (NBYN[7] * NBYN[7]) * workDevide, NULL);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[7];
        global_size[1] = NBYN[7] * NBYN[7];
        global_size[2] = workDevide;
        local_size[0] = 1024 / (NBYN[7] * NBYN[7]) / workDevide;
        local_size[1] = NBYN[7] * NBYN[7];
        local_size[2] = workDevide;
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[7], &execEvent[7]);
        CHECK_ERROR(err);






        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[8] * OUTPUT_DIM[8], w[8], 1, &execEvent[6], &memEvent[8][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[8], b[8], 1, &execEvent[6], &memEvent[8][1]);
        CHECK_ERROR(err);

        workDevide = 8;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[8]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[8]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[8]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[8] * NBYN[8] * workDevide, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3 * 1024 / (NBYN[8] * NBYN[8]) * workDevide, NULL);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[8];
        global_size[1] = NBYN[8] * NBYN[8];
        global_size[2] = workDevide;
        local_size[0] = 1024 / (NBYN[8] * NBYN[8]) / workDevide;
        local_size[1] = NBYN[8] * NBYN[8];
        local_size[2] = workDevide;
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[8], &execEvent[8]);
        CHECK_ERROR(err);




        err = clSetKernelArg(clInfo.kernel_maxpool, 0, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_maxpool, 1, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_maxpool, 2, sizeof(cl_int), &NBYN[9]);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[9];
        global_size[1] = NBYN[9] * NBYN[9];
        local_size[0] = 1024 / (NBYN[9] * NBYN[9]);
        local_size[1] = NBYN[9] * NBYN[9];
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_maxpool, 2, NULL,
            global_size, local_size,
            1, &execEvent[8], &execEvent[9]);
        CHECK_ERROR(err);





        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf2, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[10] * OUTPUT_DIM[10], w[10], 1, &execEvent[7], &memEvent[10][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf2, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[10], b[10], 1, &execEvent[7], &memEvent[10][1]);
        CHECK_ERROR(err);

        workDevide = 16;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf2);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf2);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[10]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[10]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[10]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[10] * NBYN[10] * workDevide, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3 * 1024 / (NBYN[10] * NBYN[10]) * workDevide, NULL);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[10];
        global_size[1] = NBYN[10] * NBYN[10];
        global_size[2] = workDevide;
        local_size[0] = 1024 / (NBYN[10] * NBYN[10]) / workDevide;
        local_size[1] = NBYN[10] * NBYN[10];
        local_size[2] = workDevide;
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[10], &execEvent[10]);
        CHECK_ERROR(err);






        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[11] * OUTPUT_DIM[11], w[11], 1, &execEvent[8], &memEvent[11][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[11], b[11], 1, &execEvent[8], &memEvent[11][1]);
        CHECK_ERROR(err);

        workDevide = 16;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[11]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[11]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[11]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[11] * NBYN[11] * workDevide, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3 * 1024 / (NBYN[11] * NBYN[11]) * workDevide, NULL);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[11];
        global_size[1] = NBYN[11] * NBYN[11];
        global_size[2] = workDevide;
        local_size[0] = 1024 / (NBYN[11] * NBYN[11]) / workDevide;
        local_size[1] = NBYN[11] * NBYN[11];
        local_size[2] = workDevide;
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[11], &execEvent[11]);
        CHECK_ERROR(err);





        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf2, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[12] * OUTPUT_DIM[12], w[12], 1, &execEvent[10], &memEvent[12][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf2, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[12], b[12], 1, &execEvent[10], &memEvent[12][1]);
        CHECK_ERROR(err);

        workDevide = 16;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf2);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf2);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[12]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[12]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[12]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[12] * NBYN[12] * workDevide, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3 * 1024 / (NBYN[12] * NBYN[12]) * workDevide, NULL);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[12];
        global_size[1] = NBYN[12] * NBYN[12];
        global_size[2] = workDevide;
        local_size[0] = 1024 / (NBYN[12] * NBYN[12]) / workDevide;
        local_size[1] = NBYN[12] * NBYN[12];
        local_size[2] = workDevide;
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[12], &execEvent[12]);
        CHECK_ERROR(err);





        err = clSetKernelArg(clInfo.kernel_maxpool, 0, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_maxpool, 1, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_maxpool, 2, sizeof(cl_int), &NBYN[13]);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[13];
        global_size[1] = NBYN[13] * NBYN[13];
        local_size[0] = 1024 / (NBYN[13] * NBYN[13]);
        local_size[1] = NBYN[13] * NBYN[13];
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_maxpool, 2, NULL,
            global_size, local_size,
            1, &execEvent[12], &execEvent[13]);
        CHECK_ERROR(err);






        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[14] * OUTPUT_DIM[14], w[14], 1, &execEvent[11], &memEvent[14][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[14], b[14], 1, &execEvent[11], &memEvent[14][1]);
        CHECK_ERROR(err);

        workDevide = 4;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[14]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[14]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[14]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[14] * NBYN[14] * workDevide, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3 * 1024 / (NBYN[14] * NBYN[14]) * workDevide, NULL);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[14];
        global_size[1] = NBYN[14] * NBYN[14];
        global_size[2] = workDevide;
        local_size[0] = 1024 / (NBYN[14] * NBYN[14]) / workDevide; //64
        local_size[1] = NBYN[14] * NBYN[14]; // 4
        local_size[2] = workDevide; // 4
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[14], &execEvent[14]);
        CHECK_ERROR(err);





        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf2, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[15] * OUTPUT_DIM[15], w[15], 1, &execEvent[12], &memEvent[15][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf2, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[15], b[15], 1, &execEvent[12], &memEvent[15][1]);
        CHECK_ERROR(err);

        workDevide = 4;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf2);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf2);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[15]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[15]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[15]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[15] * NBYN[15] * workDevide, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3 * 1024 / (NBYN[15] * NBYN[15]) * workDevide, NULL);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[15];
        global_size[1] = NBYN[15] * NBYN[15];
        global_size[2] = workDevide;
        local_size[0] = 1024 / (NBYN[15] * NBYN[15]) / workDevide; //64
        local_size[1] = NBYN[15] * NBYN[15]; // 4
        local_size[2] = workDevide; // 4
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[15], &execEvent[15]);
        CHECK_ERROR(err);






        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf, CL_FALSE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[16] * OUTPUT_DIM[16], w[16], 1, &execEvent[14], &memEvent[16][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[16], b[16], 1, &execEvent[14], &memEvent[16][1]);
        CHECK_ERROR(err);

        workDevide = 4;
        err = clSetKernelArg(clInfo.kernel_conv, 0, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_conv, 1, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_conv, 2, sizeof(cl_mem), &wBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 3, sizeof(cl_mem), &bBuf);
        err = clSetKernelArg(clInfo.kernel_conv, 4, sizeof(int), &INPUT_DIM[16]);
        err = clSetKernelArg(clInfo.kernel_conv, 5, sizeof(int), &OUTPUT_DIM[16]);
        err = clSetKernelArg(clInfo.kernel_conv, 6, sizeof(int), &NBYN[16]);
        err = clSetKernelArg(clInfo.kernel_conv, 7, sizeof(int), &workDevide);
        err = clSetKernelArg(clInfo.kernel_conv, 8, sizeof(float) * NBYN[16] * NBYN[16] * workDevide, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 9, sizeof(float) * 1024, NULL);
        err = clSetKernelArg(clInfo.kernel_conv, 10, sizeof(float) * 3 * 3 * 1024 / (NBYN[16] * NBYN[16]) * workDevide, NULL);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[16];
        global_size[1] = NBYN[16] * NBYN[16];
        global_size[2] = workDevide;
        local_size[0] = 1024 / (NBYN[16] * NBYN[16]) / workDevide; //64
        local_size[1] = NBYN[16] * NBYN[16]; // 4
        local_size[2] = workDevide; // 4
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_conv, 3, NULL,
            global_size, local_size,
            2, memEvent[16], &execEvent[16]);
        CHECK_ERROR(err);






        err = clSetKernelArg(clInfo.kernel_maxpool, 0, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_maxpool, 1, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_maxpool, 2, sizeof(cl_int), &NBYN[17]);
        CHECK_ERROR(err);

        global_size[0] = OUTPUT_DIM[17];
        global_size[1] = NBYN[17] * NBYN[17];
        local_size[0] = OUTPUT_DIM[17]; // OUTPUT_DIM[17]은 512이기에 global 크기인 1024보다 작다. 그래서 1024 / (NBYN[17] * NBYN[17]); 대신 드감
        local_size[1] = NBYN[17] * NBYN[17];
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_maxpool, 2, NULL,
            global_size, local_size,
            1, &execEvent[16], &execEvent[17]);
        CHECK_ERROR(err);




        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf2, CL_FALSE, 0, sizeof(float) * INPUT_DIM[18] * OUTPUT_DIM[18], w[18], 1, &execEvent[15], &memEvent[18][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf2, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[18], b[18], 1, &execEvent[15], &memEvent[18][1]);
        CHECK_ERROR(err);


        err = clSetKernelArg(clInfo.kernel_fc, 0, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_fc, 1, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_fc, 2, sizeof(cl_mem), &wBuf2);
        err = clSetKernelArg(clInfo.kernel_fc, 3, sizeof(cl_mem), &bBuf2);
        err = clSetKernelArg(clInfo.kernel_fc, 4, sizeof(float) * 1024, NULL);

        global_size[0] = INPUT_DIM[18];
        global_size[1] = OUTPUT_DIM[18];
        local_size[0] = INPUT_DIM[18];
        local_size[1] = 1024 / INPUT_DIM[18];
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_fc, 2, NULL,
            global_size, local_size,
            2, memEvent[18], &execEvent[18]);
        CHECK_ERROR(err);




        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf, CL_FALSE, 0, sizeof(float) * INPUT_DIM[19] * OUTPUT_DIM[19], w[19], 1, &execEvent[16], &memEvent[19][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[19], b[19], 1, &execEvent[16], &memEvent[19][1]);
        CHECK_ERROR(err);


        err = clSetKernelArg(clInfo.kernel_fc, 0, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_fc, 1, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_fc, 2, sizeof(cl_mem), &wBuf);
        err = clSetKernelArg(clInfo.kernel_fc, 3, sizeof(cl_mem), &bBuf);
        err = clSetKernelArg(clInfo.kernel_fc, 4, sizeof(float) * 1024, NULL);

        global_size[0] = INPUT_DIM[19];
        global_size[1] = OUTPUT_DIM[19];
        local_size[0] = INPUT_DIM[19];
        local_size[1] = 1024 / INPUT_DIM[19];
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_fc, 2, NULL,
            global_size, local_size,
            2, memEvent[19], &execEvent[19]);
        CHECK_ERROR(err);



        err = clEnqueueWriteBuffer(clInfo.memoryQueue, wBuf2, CL_FALSE, 0, sizeof(float) * INPUT_DIM[20] * OUTPUT_DIM[20], w[20], 1, &execEvent[18], &memEvent[20][0]);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clInfo.memoryQueue, bBuf2, CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[20], b[20], 1, &execEvent[18], &memEvent[20][1]);
        CHECK_ERROR(err);


        err = clSetKernelArg(clInfo.kernel_fc, 0, sizeof(cl_mem), &layerBuf[0]);
        err = clSetKernelArg(clInfo.kernel_fc, 1, sizeof(cl_mem), &layerBuf[1]);
        err = clSetKernelArg(clInfo.kernel_fc, 2, sizeof(cl_mem), &wBuf2);
        err = clSetKernelArg(clInfo.kernel_fc, 3, sizeof(cl_mem), &bBuf2);
        err = clSetKernelArg(clInfo.kernel_fc, 4, sizeof(float) * 1024, NULL);

        global_size[0] = INPUT_DIM[20];
        global_size[1] = OUTPUT_DIM[20];
        local_size[0] = INPUT_DIM[20];
        local_size[1] = 1024 / INPUT_DIM[20];
        err = clEnqueueNDRangeKernel(
            clInfo.taskQueue, clInfo.kernel_fc, 2, NULL,
            global_size, local_size,
            2, memEvent[20], &execEvent[20]);
        CHECK_ERROR(err);

        err = clEnqueueReadBuffer(clInfo.memoryQueue, layerBuf[1], CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[20], result, 1, &execEvent[20], NULL);
        CHECK_ERROR(err);


        //clFinish(clInfo.taskQueue);
        //clFinish(clInfo.memoryQueue);

        softmax(result, 10);

        labels[i] = find_max(result, 10);
        confidences[i] = result[labels[i]];
        images += 32 * 32 * 3;





    }
    end = clock();

    printf("Elapsed time: %f sec\n", (double)(end - start) / CLK_TCK);

}

