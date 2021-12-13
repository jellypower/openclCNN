#define MAX_WORK_GROUP_SIZE 1024


__kernel void convolution(__global float* input, __global float* output, __global float* filter, __constant float* bias,
	const int inDim, const int outDim, const int nbyn,
	int workDevide, __local float* sub, __local float* inter_conv, __local float* filter_tile) {

	// filter_tile의 사이즈는 workDevide*(1024/(nbyn*nbun)/workDevide)*3*3이다.
	//sub의 사이즈는 workDevide*nbyn*nbyn이다. -> sub의 용도는 inDim의 각 층층 이미지들을 타일링 하는것
	// -> 둘이서 동시에 계산을 진행하려면 타일링도 두 개를 진행해야함.

	// __local float *inter_conv 의 사이즈는 [1024/(nbyn*nbun)/workDevide][nbyn*nbyn][workDevide]; 이다.

	// 원래 convolutin 연산을 진행하면 해당 output filter의 종류에 따라서
	// 여러개의 inputDim이 합쳐져 하나의 outputDim으로 나오는데
	// 이런 outputDim의 연산결과를 workDevide만큼 여러개의 work_group으로 나누어 병렬 하는 방식 

	const int LoutNo = get_local_id(0); // outDim 에 대응한다.
	const int row = get_local_id(1)/nbyn; // y축
	const int col = get_local_id(1)%nbyn; // x축
	const int dvdNo = get_local_id(2); // workDevide에 대응한다.

	const int GoutNo = get_global_id(0); // outDim에 대응한다.

	const int LoutNoMax = get_local_size(0);

	const int nbynpow = nbyn * nbyn;

	const int inDimOffset = inDim / workDevide;
	// iinDim 개수를 workDevide로 나누면 연산울 devide한 이후
	// 하나의 PE에서 얼마의 inDim을 연산하는지에 대한 값이 나온다.

	const int workDevideOffset = MAX_WORK_GROUP_SIZE / workDevide;
	

	int x, y;

	inter_conv[ workDevideOffset* dvdNo + nbynpow* LoutNo + nbyn*row + col ] = 0;



	// outNo 의 최댓값은 1024/(nbyn*nbyn)/workDevide
	
	int t_row_start = (row == 0) ? 1 : 0;
	int t_row_end = (row == nbyn - 1) ? 2 : 3;
	int t_col_start = (col == 0) ? 1 : 0;
	int t_col_end = (col == nbyn - 1) ? 2 : 3;


	

	//if (nbynpow == 256) filter += (10 * inDim * 9);
	//else
	filter += (GoutNo * inDim * 9); // filter를 내가 구하고자하는 outDim으로 이동시키기
	
	//if (nbynpow == 256) printf("%d\n", 9 * inDimOffset * dvdNo);


	filter += 9 * inDimOffset * dvdNo;
	//if (nbynpow == 256) printf("%d\n", 9 * inDimOffset * dvdNo);
	
	//여기서 inNeuron은 몇 번째 inDim인지 와 대응한다. 
	for (int inNeuron = 0; inNeuron < inDimOffset; inNeuron++) { // workDevide만큼 inDim을 나눠서 계산
		

		if (LoutNo == 0) {
			sub[nbynpow * dvdNo +nbyn * row + col] = input[nbynpow * (inNeuron + dvdNo * inDimOffset) + nbyn * row + col];
			// work를 devide하면 sub가 2개 있어야 하는가?
			// sub 는 각 차원을 1개씩 local에 타일링 해오는 것을 의미하기에 workDevide만큼 필요
			
		}
		if (row < 3 && col < 3) {

			filter_tile[LoutNoMax * dvdNo * 9 + 9 * LoutNo + 3 * row + col] = filter[3 * row + col];

			if (nbyn < 3) { // 필터는 3*3인데 nbyn이 3*3보다 작은 2*2이면 타일링을 별도로 해줘야함.
				if(col==1)
					filter_tile[LoutNoMax * dvdNo * 9 + 9 * LoutNo + 3 * row + 2] = filter[3 * row + 2];
				if(row==1)
					filter_tile[LoutNoMax * dvdNo * 9 + 9 * LoutNo + 3 * 2 + col] = filter[3 * 2 + col];
				if(col==1 && row==1)
					filter_tile[LoutNoMax * dvdNo * 9 + 9 * LoutNo + 3 * 2 + 2] = filter[3 * 2 + 2];
			}
			
		}
		
		

		barrier(CLK_LOCAL_MEM_FENCE);
	


		
		for (int t_row = t_row_start; t_row < t_row_end; t_row++) {
			for (int t_col = t_col_start; t_col < t_col_end; t_col++) {

				y = row + t_row - 1;
				x = col + t_col - 1;

				//row 의 max 는 31 li2의 max는 31 + 32*32*3

				//inter_conv[workDevideOffset * ]

				
				inter_conv[workDevideOffset * dvdNo + nbynpow * LoutNo + nbyn * row + col]
					+= (sub[nbynpow * dvdNo + nbyn * y + x]
						* filter_tile[LoutNoMax * dvdNo * 9 + 9 * LoutNo + 3 * t_row + t_col]);
						//* filter[3 * t_row + t_col]);
			}
		}
		filter += 9;

		barrier(CLK_LOCAL_MEM_FENCE);

	}

	barrier(CLK_LOCAL_MEM_FENCE);



	if (dvdNo == 0) {
		float sum = 0;
		for (int i = 0; i < workDevide; i++) {

			sum += inter_conv[workDevideOffset * i + nbynpow * LoutNo + nbyn * row + col];
		}

		

		sum += bias[GoutNo];
		
		output[nbynpow * GoutNo + nbyn * row + col] =
			sum > 0 ? sum : 0;
		
	}

}

__kernel void max_pooling(__global float* input, __global float* output, const int nbyn) {
	// input은 [inDim][nbyn*nbyn]
	// out은 [inDim][(nbyn/2)*(nbyn/2)]

	// 그러면 global_size는 [inDim][(nbyn/2)*(nbyn/2)]
	// local_size는 [1024/(nbyn/2)(nbyn/2)][(nbyn/2)*(nbyn/2)]

	//const int nbyn의 값은 input이미지의 절반이다.


	int gi0 = get_global_id(0);
	int row = get_global_id(1)/nbyn;
	int col = get_global_id(1)%nbyn;

	float max = 0;
	float tmp;

	input += ((nbyn*2) * (nbyn*2)) * gi0;
	output += nbyn * nbyn * gi0;


	//--------------------------------

	tmp = input[(row * 2 + 0) * (nbyn*2) + (col * 2 + 0)];
	max= tmp;
	tmp = input[(row * 2 + 0) * (nbyn * 2) + (col * 2 + 1)];
	if (tmp > max) max = tmp;
	tmp = input[(row * 2 + 1) * (nbyn * 2) + (col * 2 + 0)];
	if (tmp > max) max = tmp;
	tmp = input[(row * 2 + 1) * (nbyn * 2) + (col * 2 + 1)];
	if (tmp > max) max = tmp;

	output[row * nbyn + col] = max;


	//if (gi0 == 0) printf("%d\n", output[row * nbyn + col]);
	
}

/*__kernel void convolution(__global float* input, __global float* output, __global float* filter, __constant float* bias,
	const int inDim, const int outDim, const int nbyn,
	int workDevide, __local float* sub, __local float* inter_conv, __local float* filter_tile)
	*/


//global size 는 [INPUTDIM][OUTPUTDIM]
//local size는 [INPUTDIM][1024/INPUTDIM]

__kernel void fully_connected(__global float *input, __global float *output, __global float* filter, __constant float* bias,
	__local float* subSum) {
	
	const int col = get_global_id(0); // fc_layer에서 filter는 가로 INPUT_DIM, 세로 OUTPUT_DIM만큼 있다.
	const int row = get_global_id(1); 

	const int local_row = get_local_id(1);

	const int colSize = get_global_size(0);

	
	subSum[local_row * colSize + col] = input[col] * filter[row * colSize + col];

	/*if (row == 0) printf("subSum[%d][%d]: %f = %f * %f\n", row, col, subSum[local_row * colSize + col],
		input[col],
		filter[row * colSize + col]
	);*/

	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int p = colSize / 2; p >= 1; p = p >> 1) {

		if (col < p) subSum[local_row * colSize + col] += subSum[local_row * colSize + col + p];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (col == 0) {
		float result = subSum[local_row * colSize] + bias[row];
		output[row] = result > 0 ? result : 0;

	}


}