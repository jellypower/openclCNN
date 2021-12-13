#pragma once



void cnn_seq(float* images, float* network, int* labels, float* confidences, int num_of_image);
void compare(const char* filename, int num_of_image);


struct openCLInfo openCLInit(const int PLAT_NO, const int DEV_NO);
void cnn_init();
void cnn(float* images, float** network, int* labels, float* confidences, int num_images);