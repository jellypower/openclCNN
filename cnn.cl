#define MAX_WORK_GROUP_SIZE 1024


__kernel void convolution(__global float* input, __global float* output, __global float* filter, __constant float* bias,
	const int inDim, const int outDim, const int nbyn,
	int workDevide, __local float* sub, __local float* inter_conv, __local float* filter_tile) {

	// filter_tile�� ������� workDevide*(1024/(nbyn*nbun)/workDevide)*3*3�̴�.
	//sub�� ������� workDevide*nbyn*nbyn�̴�. -> sub�� �뵵�� inDim�� �� ���� �̹������� Ÿ�ϸ� �ϴ°�
	// -> ���̼� ���ÿ� ����� �����Ϸ��� Ÿ�ϸ��� �� ���� �����ؾ���.

	// __local float *inter_conv �� ������� [1024/(nbyn*nbun)/workDevide][nbyn*nbyn][workDevide]; �̴�.

	// ���� convolutin ������ �����ϸ� �ش� output filter�� ������ ����
	// �������� inputDim�� ������ �ϳ��� outputDim���� �����µ�
	// �̷� outputDim�� �������� workDevide��ŭ �������� work_group���� ������ ���� �ϴ� ��� 

	const int LoutNo = get_local_id(0); // outDim �� �����Ѵ�.
	const int row = get_local_id(1)/nbyn; // y��
	const int col = get_local_id(1)%nbyn; // x��
	const int dvdNo = get_local_id(2); // workDevide�� �����Ѵ�.

	const int GoutNo = get_global_id(0); // outDim�� �����Ѵ�.

	const int LoutNoMax = get_local_size(0);

	const int nbynpow = nbyn * nbyn;

	const int inDimOffset = inDim / workDevide;
	// iinDim ������ workDevide�� ������ ����� devide�� ����
	// �ϳ��� PE���� ���� inDim�� �����ϴ����� ���� ���� ���´�.

	const int workDevideOffset = MAX_WORK_GROUP_SIZE / workDevide;
	

	int x, y;

	inter_conv[ workDevideOffset* dvdNo + nbynpow* LoutNo + nbyn*row + col ] = 0;



	// outNo �� �ִ��� 1024/(nbyn*nbyn)/workDevide
	
	int t_row_start = (row == 0) ? 1 : 0;
	int t_row_end = (row == nbyn - 1) ? 2 : 3;
	int t_col_start = (col == 0) ? 1 : 0;
	int t_col_end = (col == nbyn - 1) ? 2 : 3;


	

	//if (nbynpow == 256) filter += (10 * inDim * 9);
	//else
	filter += (GoutNo * inDim * 9); // filter�� ���� ���ϰ����ϴ� outDim���� �̵���Ű��
	
	//if (nbynpow == 256) printf("%d\n", 9 * inDimOffset * dvdNo);


	filter += 9 * inDimOffset * dvdNo;
	//if (nbynpow == 256) printf("%d\n", 9 * inDimOffset * dvdNo);
	
	//���⼭ inNeuron�� �� ��° inDim���� �� �����Ѵ�. 
	for (int inNeuron = 0; inNeuron < inDimOffset; inNeuron++) { // workDevide��ŭ inDim�� ������ ���
		

		if (LoutNo == 0) {
			sub[nbynpow * dvdNo +nbyn * row + col] = input[nbynpow * (inNeuron + dvdNo * inDimOffset) + nbyn * row + col];
			// work�� devide�ϸ� sub�� 2�� �־�� �ϴ°�?
			// sub �� �� ������ 1���� local�� Ÿ�ϸ� �ؿ��� ���� �ǹ��ϱ⿡ workDevide��ŭ �ʿ�
			
		}
		if (row < 3 && col < 3) {

			filter_tile[LoutNoMax * dvdNo * 9 + 9 * LoutNo + 3 * row + col] = filter[3 * row + col];

			if (nbyn < 3) { // ���ʹ� 3*3�ε� nbyn�� 3*3���� ���� 2*2�̸� Ÿ�ϸ��� ������ �������.
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

				//row �� max �� 31 li2�� max�� 31 + 32*32*3

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
	// input�� [inDim][nbyn*nbyn]
	// out�� [inDim][(nbyn/2)*(nbyn/2)]

	// �׷��� global_size�� [inDim][(nbyn/2)*(nbyn/2)]
	// local_size�� [1024/(nbyn/2)(nbyn/2)][(nbyn/2)*(nbyn/2)]

	//const int nbyn�� ���� input�̹����� �����̴�.


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


//global size �� [INPUTDIM][OUTPUTDIM]
//local size�� [INPUTDIM][1024/INPUTDIM]

__kernel void fully_connected(__global float *input, __global float *output, __global float* filter, __constant float* bias,
	__local float* subSum) {
	
	const int col = get_global_id(0); // fc_layer���� filter�� ���� INPUT_DIM, ���� OUTPUT_DIM��ŭ �ִ�.
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