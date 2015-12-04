#ifndef XMLPARSE_H
#define XMLPARSE_H

#include "types.h"

#define STAGENUM 25

int parseClassifier(char *filename, int& stagesNum, stageMeta_t*& stagesMeta, stage_t**& stages, feature_t*& features);

#endif XMLPARSE_H