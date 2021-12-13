#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <windows.h>
#include <math.h>
#include <time.h>
#include <direct.h>


extern const int INPUT_DIM[] = {
	3, 64, //맨 처음 input은 RGB 3차원임
	64,

	64,128,
	128,

	128, 256, 256,
	256,

	256, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	512
};

extern const int OUTPUT_DIM[] = {
	64, 64,
	64,

	128, 128,
	128,

	256, 256, 256,
	256,

	512, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	10 // 최종 결과는 10가지 중 1개임 main.c > CLASS_NAME
};

extern const int NBYN[] = {
	32, 32,
	16, // max pooling이 진행될 때마다 1/2로 줄어든다.

	16, 16,
	8,

	8, 8, 8,
	4,

	4, 4, 4,
	2,

	2, 2, 2,
	1,

	1,
	1,
	1
};



extern const char* CLASS_NAME[];

static void convolution(float* inputs, float* outputs, float* filter, float* biases, int inDim, int outDim, int nbyn) {
	// output 변수는  가로->세로->dim 순으로 값들이 저장된다.
	// input 변수도 가로->세로->dim 순으로 값들이 저장된다.

	memset(outputs, 0, nbyn * nbyn * outDim * sizeof(float));
	int x = 0, y = 0;
	int offset = nbyn * nbyn; //input 이미지의 크기
	float sum = 0, temp;
	float* input, * output;
	for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) { //output에서 몇 개의 층이 나오는지
		input = inputs; // 다시 해당 사진의 처음으로 회귀한 이후 다음 필터로 넘어가서 진행
		for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) { //input에서 각 layer에 맞는 필터를 이용해 여러 dim을 하나로 합치기
			output = outputs;
			for (int row = 0; row < nbyn; ++row) {
				for (int col = 0; col < nbyn; ++col) { // 사진의 x, y크기
					sum = 0;
					for (int fRow = 0; fRow < 3; ++fRow) {
						for (int fCol = 0; fCol < 3; ++fCol) { // 각 필터는 3*3짜리

							x = col + fCol - 1;
							y = row + fRow - 1;

							if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
								sum += input[nbyn * y + x] * filter[3 * fRow + fCol]; //각 필터별 가중치 구하기
							}

						}
					}
					*(output++) += sum; //가중치의 합을 output에 저장
				}
			}
			filter += 9; //다음 필터로 넘어가기 -> 가령, RGB사진이면 필터의 역할에 따라 R, G, B 3개의 필터가 필요함 
			input += offset; //해당 사진의 다음 차원으로 넘어가기 ex) R->G->B

		}
		for (int i = 0; i < offset; ++i) {
			(*outputs) = (*outputs) + (*biases); // 가중치 더해주기
			if (*outputs < 0) (*outputs) = 0;	//ReLU
			outputs++; // 다음 output으로 넘어가기 -> 파라미터로 들어오는 outputs 자체의 포인터 값을 증가시켜 결국 해당 for문을 모두 돌면 다음 dimension으로 넘어감
		}
		++biases; //다음 가중치로 넘어가기
	}

}


//각 PE하나에 픽셀 하나를 할당하는 개념으로 가자
//그리고, local메모리는 계산하려는 전체 이미지를 잘라서 최대한 local memory에 잘라서 우겨넣자
//가령, 이미지 사이즈가 1기가인데 local 메모리가 250메가바이트면 4개로 잘라서 local memory에 넣는거임
//


//결국 픽셀 하나당 3*3*inputDim의 연산은 seq하게 진행돼야함 -> seq하게 진행한 이후 나중에 값을 합해주는 방법도 있음


//outDim은 종속성을 지님
//픽셀 하나하나는 종속성을 지님  =========================> 결론, 우선 1단계로 outDim이랑 픽셀 하나하나의 종속성을 지키며 병렬연산을 진행하자
// ========================================================> 우선 global_size는 outputDim*사진픽셀수(=nbyn*nbyn) 로 진행하자.
// ========================================================> 그러면 병렬 연산 가능한 1개 유닛당 연산시간은 3*3*inDim이다. ====> 1차 전략 완료
// ========================================================> 즉, 각 private이 필요한 local memory는 3*3*inDim이다.
// ========================================================> work_group_size는 최대한 256=16*16에 맞추는게 좋다.
// ========================================================> 그리고, 각 PE는 __local float subA[256]; 에다가 타일링 진행.
// ========================================================> 타일링 하다보면 nbyn사이즈에 따라 하나의 배열에 여러 dim의 값이 들어올 수 있는데 그
// ========================================================> 그런 경우에는 어디까지가 해당 배열의 값인지 잘 계산하도록 로직을 구성하자.
// ========================================================> 필터는 __constant로 받자 -> 어차피 필터는 변하면 안되니까

// ========================================================> convolution을 여러번 진행할텐데 convolution 진행하면서 동시에 weight(=filter), bias 를 각 레이어에 맞게 나눠서 GPU에 enque해주자.
// ========================================================> dim 정보와 nbyn정보는 파라미터로 전달


// ========================================================> 타일링 진행하는 방법
//만약 dim=3이라면 global(=work_item 총 개수) 사이즈는 nbyn*nbyn*outDim
//만약 dim=3이라면 타일링 사이즈(=work_group_size)는 __local float subA[nbyn][nbyn][(work_group_max)/(nbyn*nbyn)]
//만약 dim=3인데 nbyn>16 이라면 subA[16+2][16+2][1]  ===> nbyn>16이면 멱수로 봤을 때, 32인데 3*3타일을 계산할 때, 마지막 원소는 최대 local_id의 범위인 16*16을 뛰어넘는 값까지 같이 convolution해야하기 때문에
	//큰 for문
	//타일 사이즈로 global_work_item개수를 나눈 만큼 for문을 반복한다. 즉, work_group 개수(=/=work_group_size)만큼 반복한다.
		//병렬으로 work_group의 타일을 대입한다. -> nbyn*nbyn*(work_group_max)/(nbyn*nbyn) 만큼의 타일이 병렬 대입됨
		//이후 barrier를 친다.
		//작은 for문
		//각 work_group의 원소들에 대해서 3*3*(work_group_max)/(nbyn*nbyn) 만큼 반복문을 돈다.
			//inter_sum을 계산한다.
		//작은 for문 종료
		//이후 barrier를 친다.
	//큰 for문 종료
	//글로벌 메모리에 값들을 업데이트 해준다.

// ========================================================> convolution 처음에, INPUT_DIM=3 인거 예외처리 하자.
// ===========> work deivde 계수를 따로 만들어서 global_size를 늘려보자
// ===========> global[outDim][nbyn][nbyn][workDeivde]  =================> inDim으로부터 나오는 work size를 workDevide만큼 나눠보자.
// ===========> global[outDim][nbyn][nbyn][workDeivde]  =================> 예를 들어 workDevide=2이면 inDim=128 이면 64, 64씩 나눠서 workDevide=0, workDevide=1에 각각 저장 
//  
// ===========> local[1024/(nbyn*nbyn)/workDevide][nbyn][nbyn][workDevide]
// ===========> 차례대로 [outputDim][nbyn][nbyn][workDevide]

// __sub[1024/(nbyn*nbyn)/workDevide][nbyn+2][nbyn+2]


// global[outDim][nbyn][nbyn]
// local[1024/(nbyn*nbyn)][nbyn][nbyn]
// PE 의 operation은 3*3*inDim



//하나의 PE가 혼자 하는 부분과 협력 하는 부분으로 나누면 좀 더 생각하기가 쉽다.






static void max_pooling(float* input, float* output, int DIM, int nbyn) {
	float max,temp;
	int n, row, col, x, y;
	for (n = 0; n < DIM; ++n) {
		for (row = 0; row < nbyn; row += 2) {
			for (col = 0; col < nbyn; col += 2) {
				//max = -FLT_MAX;
				max = 0;
				for (y = 0; y < 2; ++y) {
					for (x = 0; x < 2; ++x) {
						temp = input[nbyn * (row + y) + col + x];
						if (max < temp) max = temp;
					}
				}
				*(output++) = max;
			}
		}
		input += nbyn * nbyn;
	}
} //마지막 convolution과 maxpooling 과정을 하나로 합쳐 메모리에 write하는 과정을 1/4로 줄이는 것을 목표로 하겠다.


void fc_layer(float* input, float* output, float* weights, float* biases, int inDim, int outDim) {
	float sum;
	for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) {
		sum = 0;
		for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
			sum += input[inNeuron] * (*weights++);
		}
		sum += biases[outNeuron];
		if (sum > 0) output[outNeuron] = sum;	//ReLU
		else output[outNeuron] = 0;
	}
} //fc_layer는 결국 행렬곱셈 연산을 진행하는 형태!

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
		input[i] = exp(input[i] - max) / (sum+1e-7);
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



void cnn_seq(float* images, float* network, int* labels, float* confidences, int num_of_image) {
	
	float* w[21];
	float* b[21];
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


	// allocate memory for layer
	float* layer[21];
	for (int i = 0; i < 21; ++i) {
		layer[i] = (float*)malloc(sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i]);
		if (layer[i] == NULL) {
			perror("malloc error");
		}
	}

	time_t start, end;
	start = clock();
	// run network
	for (int i = 0; i < num_of_image; ++i) {
		convolution(images, layer[0], w[0], b[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
		convolution(layer[0], layer[1], w[1], b[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
		max_pooling(layer[1], layer[2], INPUT_DIM[2], NBYN[2] * 2);

		convolution(layer[2], layer[3], w[3], b[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
		convolution(layer[3], layer[4], w[4], b[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
		max_pooling(layer[4], layer[5], INPUT_DIM[5], NBYN[5] * 2);

		convolution(layer[5], layer[6], w[6], b[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
		convolution(layer[6], layer[7], w[7], b[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
		convolution(layer[7], layer[8], w[8], b[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
		max_pooling(layer[8], layer[9], INPUT_DIM[9], NBYN[9] * 2);

		convolution(layer[9], layer[10], w[10], b[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
		convolution(layer[10], layer[11], w[11], b[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
		convolution(layer[11], layer[12], w[12], b[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
		max_pooling(layer[12], layer[13], INPUT_DIM[13], NBYN[13] * 2);

		convolution(layer[13], layer[14], w[14], b[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
		convolution(layer[14], layer[15], w[15], b[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
		convolution(layer[15], layer[16], w[16], b[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
		max_pooling(layer[16], layer[17], INPUT_DIM[17], NBYN[17] * 2);

		fc_layer(layer[17], layer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18]);
		fc_layer(layer[18], layer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19]);
		fc_layer(layer[19], layer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20]);

		softmax(layer[20], 10);

		labels[i] = find_max(layer[20], 10);
		confidences[i] = layer[20][labels[i]];
		images += 32 * 32 * 3;
	}
	end = clock();
	printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);

	for (int i = 0; i < 21; ++i) {
		free(layer[i]);
	}
}







//폐기된 방안들


// ========================================================> 아이템 개수 늘리기
// =======================> global 개수를 inDim*nbyn*nbyn*outDim
// =======================> work_item 개수를 [(work_group_max)/(nbyn*nbyn)][nbyn][nbyn][1]개 -> [inDim][nbyn][nbyn][1] 는 틀렸고
// =======================>(work_group_max)/(nbyn*nbyn) 부분을 reduction으로 계산하기





// =============================> 다시 생각해보니 global을 [inDim][nbyn*nbyn*outDim] 으로 하고
// =============================> local을 [inDim][1024/inDim] 으로 하면 되지 않나?
// =============================> inDim 부분을 reduction 하면 된다.
// =============================> output[global_id1] = local[0][local_id1] => convol계산하고 reduction까지 한 이후의 위치
// =============================> 픽셀 하나를 집어낼 때 local[local_id0] = input[global_id1%(nbyn*nbyn)] * filter[local_id0 번째];
// =============================> 픽셀 하나를 집고 local[local_id0]을 inDim 크기만큼 reduction 진행
// =============================> 타일링은 


// input[ + ]

// global[inDim][nbyn*nbyn][outDim]
// local[inDim][1024/(inDim)][1]  --> 이게 제일 좋은듯
// local 메모리 사이즈는 -> local[work_group_size]
// local 메모리는 2개 ㅣ_conv[1024] 랑 l_tile[3*3]
// l_tile[global_id0 % 9] = input[global_id0 * (nbyn*nbyn) + global_id1]
// output[global_id2 * (nbyn*nbyn) + global_id1] = local[]
// 

// for문의 바깥쪽일수록 병렬실행되기 좋다.
// 1. global_size = for문의 가장 바깥쪽의 outDim
// 2, global_size = global_size * for문 안쪽의 inDim 으로 지정
// 3. local_size = 최대인 256
// 4. 각 pe에는 3*3짜리 필터 하나씩 할당
// 5. local_memory에는 
// 6. 


// 중요 -> work_group 하나를 work_size로 채우기
// 결국 input img의 크기는 nbyn*nbyn*inputDim이다.
