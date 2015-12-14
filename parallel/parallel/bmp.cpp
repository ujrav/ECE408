#include "bmp.h"

using namespace std;

unsigned char* readBMP(char* filename, int &width, int &height)
{
	int i, j;
	FILE* f = fopen(filename, "rb");
	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	unsigned char red, green, blue;

	// extract image height and width from header
	width = *(int*)&info[18];
	height = *(int*)&info[22];

	int size = 3 * width * height;
	unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
	//unsigned char* flip = new unsigned char[size];
	fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);

	for (j = 0; j < height / 2; j++){
		for (i = 0; i < width; i++){
			red = data[3 * (i + j*width) + 2];
			green = data[3 * (i + j*width) + 1];
			blue = data[3 * (i + j*width)];

			data[3 * (i + j*width)] = data[3 * (i + (height - j - 1)*width) + 2];
			data[3 * (i + j*width) + 1] = data[3 * (i + (height - j - 1)*width) + 1];
			data[3 * (i + j*width) + 2] = data[3 * (i + (height - j - 1)*width)];

			data[3 * (i + (height - j - 1)*width)] = red;
			data[3 * (i + (height - j - 1)*width) + 1] = green;
			data[3 * (i + (height - j - 1)*width) + 2] = blue;

		}
	}

	return data;
}

void writeBMP(char* filename, unsigned char *data, int width, int height){
	FILE *fp;
	int i, j;
	unsigned char red, green, blue;
	unsigned char header[54] = { 0 };
	unsigned char* output = new unsigned char[3 * width*height];
	fp = fopen(filename, "wb");
	header[0] = 'B';
	header[1] = 'M';
	*(int*)&header[2] = 54 + width * height * 3;
	*(int*)&header[0xA] = 54;
	*(int*)&header[0xE] = 40;

	*(int*)&header[18] = width;
	*(int*)&header[22] = height;

	header[0x1A] = 1;
	header[0x1C] = 24;
	header[0x22] = 3 * width * height;
	fwrite(header, 1, 54, fp);
	for (i = 0; i < width; i++){
		for (j = 0; j < height / 2; j++){
			red = data[3 * (i + j*width)];
			green = data[3 * (i + j*width) + 1];
			blue = data[3 * (i + j*width) + 2];

			output[3 * (i + j*width) + 2] = data[3 * (i + (height - j - 1)*width)];
			output[3 * (i + j*width) + 1] = data[3 * (i + (height - j - 1)*width) + 1];
			output[3 * (i + j*width)] = data[3 * (i + (height - j - 1)*width) + 2];


			output[3 * (i + (height - j - 1)*width)] = blue;
			output[3 * (i + (height - j - 1)*width) + 1] = green;
			output[3 * (i + (height - j - 1)*width) + 2] = red;
		}
	}
	fwrite(output, 1, 3 * width*height, fp);
	fclose(fp);
}

void convertGrayScale(unsigned char *inputImg, float *outputImg, int imgWidth, int imgHeight)
{
	for (int i = 0; i < imgWidth*imgHeight; ++i){
		outputImg[i] = (0.2989f*((float)inputImg[3 * i]) + 0.5870f*((float)inputImg[3 * i + 1]) + 0.1140f*((float)inputImg[3 * i + 2])) / (255.0f); // in windows stored as BGR
	}
}