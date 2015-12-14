#ifndef XMLPARSE_H
#define XMLPARSE_H

#include "types.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "RapidXML\rapidxml.hpp"
#include "RapidXML\rapidxml_utils.hpp"

int parseClassifier(char *filename, int& stagesNum, stageMeta_t*& stagesMeta, stage_t**& stages, feature_t*& features);
int parseClassifierFlat(char *filename, int& stageNum, int& featureNum, stageMeta_t*& stagesMeta, stage_t*& stagesFlat, feature_t*& features);

#endif XMLPARSE_H