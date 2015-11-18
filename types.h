#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

struct stage_t{
	int featureIdx;
	float threshold;
	float leftWeight;
	float rightWeight;
};

struct stageMeta_t{
	uint8_t size;
	float threshold;
};

struct rect_t{
	uint8_t x;
	uint8_t y;
	uint8_t w;
	uint8_t h;
	int8_t weight;
};

struct feature_t{
	rect_t black;
	rect_t white;
};



#endif