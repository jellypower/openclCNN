<p align="center">
<img src="https://github.com/jellypower/PublicImageDataBase/blob/main/Portfolio/OpenCLCNN/OpenCLBanner.png" width="400"/>
</p>


> 2021년도 2학기, 세종대학교 멀티코어 프로그래밍 수업 과정에서 진행한 GPGPU를 통한 CNN네트워크 최적화 프로젝트
> 

# 프로젝트 소개

- 주제: OpenCL을 활용한 CNN네트워크 최적화
- 플랫폼: x86(64), OpenCL, Windows, Nvidia Geforce GTX1050
- 프로젝트 내용:
    - VGGNET기반 CNN 네트워크 최적화
    - 내부의 주요 레이어인 Convolution, Maxpool, Fullyconnected 를 GPU연산을 통해 최적화

# 병렬화 전략

## 메모리 전략

- 런타임 계산 이전에 GPU버퍼를 생성하고 올려 런타임중 메모리 버퍼 생성, 전송 방지

```cpp
void cnn_init(float** network, float** w, float** b, cl_mem* layerBuf, struct openCLInfo *cl) {
.
.
.
    for (int i = 0; i < 17; ++i) { // Convolution Layer buffers
        if (i == 2 || i == 5 || i == 9 || i == 13) i++;	// Skip for Maxpooling layer

        cl->wBuf[i] = clCreateBuffer(cl->context, CL_MEM_READ_ONLY, sizeof(float) * 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i], NULL, &err);
        CHECK_ERROR(err);

        cl->bBuf[i] = clCreateBuffer(cl->context, CL_MEM_READ_ONLY, sizeof(float) * OUTPUT_DIM[i], NULL, &err);
        CHECK_ERROR(err);

        err = clEnqueueWriteBuffer(cl->memoryQueue, cl->wBuf[i], CL_TRUE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i], w[i], 0, NULL, NULL);
        CHECK_ERROR(err);

        err = clEnqueueWriteBuffer(cl->memoryQueue, cl->bBuf[i], CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[i], b[i], 0, NULL, NULL);
        CHECK_ERROR(err);
    }
.
.
.
}
```

소스코드: https://github.com/jellypower/openclCNN/blob/master/cnn_opencl.c

---

- 메모리용 커맨드큐와 작업용 커맨드 큐를 분리하여 System과 GPU사이에서 발생하는 작업들과 GPU내에서만 발생하는 작업들을 분리
- 메모리 전송 동작과 GPU연산간의 커맨트 큐를 분리하여 비동기 작업들이 동시에 진행되도록 실행속도를 높임

```cpp
cl_command_queue taskQueue;
cl_command_queue memoryQueue;
.
.
.
		context = clCreateContext(NULL, 1, &devices[DEV_NO], NULL, NULL, &err);
    CHECK_ERROR(err);

    taskQueue = clCreateCommandQueue(context, devices[DEV_NO], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
        | CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);

    memoryQueue = clCreateCommandQueue(context, devices[DEV_NO], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
        | CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);
.
.
.
```

소스코드: https://github.com/jellypower/openclCNN/blob/master/cnn_opencl.c

## Convolution Layer

- Convolution레이어의 이미지처리를 CPU에서 진행할 경우 연산을 
`imgXPixel * imgYPixel * colorDimension * filterXPixel * filterYPixel * filterDimension`
횟수만큼 반복해야 한다.
- 연산 과정에서 **출력 이미지**의 한 픽셀, 한 픽셀 계산을 병렬화.

<p align="center">
<img src="https://github.com/jellypower/PublicImageDataBase/blob/main/Portfolio/OpenCLCNN/convolution.png" width="600"/>
</p>

---

- 하나의 **출력 이미지**의 한 픽셀, 한 픽셀의 연산과정을 2개의 `Processing Element`에 분담하고 최종 결과 출력부에서 프로세서의 연산을 합산하여 연산의 병렬성을 가속


<p align="center">
<img src="https://github.com/jellypower/PublicImageDataBase/blob/main/Portfolio/OpenCLCNN/convolutionworkdevide.png" width="600"/>
</p>

---

- 중간 합산결과를 GPU의 VRAM에 저장하는 것이 아닌 GPU칩셋 내의 SRAM에 저장하여 가속화

```cpp
__kernel void convolution(__global float* input, __global float* output, __global float* filter, __constant float* bias,
	const int inDim, const int outDim, const int nbyn,
	int workDevide,

 __local float* inter_conv
		// 한 픽셀, 한 픽셀의 연산과정을 2개의 Processing Element에 분담했고
		// 분담된 개별 합산결과를 SRAM(__local 메모리)에 저장

) {
.
.
.
	for (int inNeuron = 0; inNeuron < inDimOffset; inNeuron++) { // workDevide만큼 inDim을 나눠서 계산
		
		for (int t_row = t_row_start; t_row < t_row_end; t_row++) {
			for (int t_col = t_col_start; t_col < t_col_end; t_col++) {
.
.
.

				inter_conv[workDevideOffset * dvdNo + nbynpow * LoutNo + nbyn * row + col]
					+=(input[nbynpow * (inNeuron + dvdNo * inDimOffset) + nbyn * y + x]
					* filter[3 * t_row + t_col]);
					// 중간 계산결과를 SRAM내의 메모리에 저장

			}
		}
		filter += 9;
	}

	barrier(CLK_LOCAL_MEM_FENCE); // 배리어로 작업들을 동기화

	for (int p = workDevide / 2; p >= 1; p = p >> 1) {

		if (dvdNo < p) inter_conv[workDevideOffset * dvdNo + nbynpow * LoutNo + nbyn * row + col]
			+= inter_conv[workDevideOffset * (dvdNo+p) + nbynpow * LoutNo + nbyn * row + col];

		barrier(CLK_LOCAL_MEM_FENCE);
		// 한 픽셀, 한 픽셀의 연산과정이 2개의 Processing Element에 분리돼서
		// SRAM에 저장된 값을 최종적으로 동기화하며 합산.
	}
	
.
.
.
}
```

소스코드: https://github.com/jellypower/openclCNN/blob/master/cnn.cl

## Max pooling Layer

- 각 입력에서 들어오는 이미지의 픽셀들을 2*2로 분할한 이후 값이 가장 큰 픽셀의 값을 샘플링

<p align="center">
<img src="https://github.com/jellypower/PublicImageDataBase/blob/main/Portfolio/OpenCLCNN/maxpooling.png" width="400"/>
</p>


---

- 최대값을 저장하는 과정에서의 연산 또한 레지스터에 저장되는 로컬변수를 활용하여 메모리 사용에 대한 부담을 줄임

```cpp
__kernel void max_pooling(__global float* input, __global float* output, const int nbyn) {

	float max = 0;
.
.
.
	tmp = input[(row * 2 + 0) * (nbyn*2) + (col * 2 + 0)];
	max= tmp;
	tmp = input[(row * 2 + 0) * (nbyn * 2) + (col * 2 + 1)];
	if (tmp > max) max = tmp;
	tmp = input[(row * 2 + 1) * (nbyn * 2) + (col * 2 + 0)];
	if (tmp > max) max = tmp;
	tmp = input[(row * 2 + 1) * (nbyn * 2) + (col * 2 + 1)];
	if (tmp > max) max = tmp;

	output[row * nbyn + col] = max;
.
.
.	
}
```

소스코드: https://github.com/jellypower/openclCNN/blob/master/cnn.cl

## Fully connected Layer

- 네트워크의 마지막에서 이미지에 행렬곱을 진행해 최종 결과를 출력하는 레이어
- 행렬의 각 열과 벡터의 곱셈을 곱하는 과정에서,
    - 단일 스칼라값의 곱셈 하나하나를 `Processing Element`에 할당하여 계산
    - 계산된 결과는 GPU의 reduction sum을 활용하여 합산하는 방식으로 행렬곱 계산

<p align="center">
<img src="https://github.com/jellypower/PublicImageDataBase/blob/main/Portfolio/OpenCLCNN/fullyconnected.png" width="300"/>
</p>

```cpp
__kernel void fully_connected(__global float *input, __global float *output, __global float* filter, __constant float* bias,
	__local float* subSum) {
	
	const int col = get_global_id(0);
	const int row = get_global_id(1); 
	const int local_row = get_local_id(1);
	const int colSize = get_global_size(0);

	
	subSum[local_row * colSize + col] = input[col] * filter[row * colSize + col];
	// 각 요소의 곱셈은 1번만 수행된다.

	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int p = colSize / 2; p >= 1; p = p >> 1) {

		if (col < p) subSum[local_row * colSize + col] += subSum[local_row * colSize + col + p];
		barrier(CLK_LOCAL_MEM_FENCE);
		// 최종적으로 reduction sum을 통해 값을 출력
	}

	if (col == 0) {
		float result = subSum[local_row * colSize] + bias[row];
		output[row] = result > 0 ? result : 0;
	}
	// 최종 결과값은 ReLu Layer를 통과
	// ReLu Layer의 경우 계산이 간단하고 Fully connected Layer 연산이 끝나고
	// 항상 진행되기에 하나의 커널코드에 결합

}
```

소스코드: https://github.com/jellypower/openclCNN/blob/master/cnn.cl

# 최종결과

## 속도변화

<p align="center">
<img src="https://github.com/jellypower/PublicImageDataBase/blob/main/Portfolio/OpenCLCNN/OptimizationResult.png" width="600"/>
</p>

## 프로파일링

- 네트워크 연산의 대부분은 convolution layer에서 소모되는 것을 확인할 수 있다.

<p align="center">
<img src="https://github.com/jellypower/PublicImageDataBase/blob/main/Portfolio/OpenCLCNN/ProfilingResult.png" width="400"/>
</p>


# 프로젝트 피드백

1. 딥러닝의 이미지처리는 보통 img2col이라는 사전 작업을 통해 이미지 데이터의 레이아웃을 행렬형태 로 전환하여 연산과정을 가속화한다. img2col과정을 거치지 않음으로써 메모리 소모를 아끼긴 했지만 범용적인 처리과정을 거치지 않아서 레이어 배치에 있어 유연성이 부과되지 못한 점이 아쉽다.
2. 프로젝트 진행당시 GPU에도 벡터연산을 가속하는 SIMD명령어가 따로 있는것을 당시에 알지못해 사용하지 않았었다. GPU의 특성을 더욱 잘 이해하고 있었다면 더욱 빠른 연산결과를 낼 수 있었다는 점이 아쉬움으로 남는다.
