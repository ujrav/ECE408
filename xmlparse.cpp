#include <stdio.h>
#include <iostream>
#include "RapidXML\rapidxml.hpp"
#include "RapidXML\rapidxml_utils.hpp"
#include "types.h"

#define STAGENUM 25

using namespace std;
using namespace rapidxml;

int parseStages(xml_node<> *root, stage_t **stages, stageMeta_t *stagesMeta);
int parseFeatures(xml_node<> *root, feature_t *features, int& featureNum);
int countFeatures(xml_node<> *root);

int parseClassifier(char *filename){
	int i = 0;
	int featureNum;
	stageMeta_t *stagesMeta;
	stage_t **stages;
	feature_t *features;

	stagesMeta = new stageMeta_t[STAGENUM]; // not arbitrary. fix later?
	stages = new stage_t*[STAGENUM];

	file<> xmlfile(filename); // default template is char
	xml_document<> doc;
	doc.parse<0>(xmlfile.data());

	xml_node<> *node = doc.first_node()->first_node()->first_node("stages");
	parseStages(node, stages, stagesMeta);
	
	node = node->next_sibling("features");
	featureNum = countFeatures(node);
	features = new feature_t[featureNum];
	parseFeatures(node, features, featureNum);

	return 0;
}

int parseStages(xml_node<> *root, stage_t **stages, stageMeta_t*stagesMeta){
	xml_node<> *featureNode;
	xml_node<> *node;
	xml_node<> *classifierNode;
	xml_node<> *internalStageNode;
	int i = 0;
	int j = 0;
	int a;
	int temp;
	float tempF;
	char *end;

	if (root == NULL){
		return -1;
	}

	for (featureNode = root->first_node(); featureNode; featureNode = featureNode->next_sibling()){
		node = featureNode->first_node("maxWeakCount");
		temp = strtol(node->value(), NULL, 10);
		stagesMeta[i].size = temp;
		node = featureNode->first_node("stageThreshold");
		tempF = strtof(node->value(), NULL);
		stagesMeta[i].threshold = tempF;

		stages[i] = new stage_t[temp];
		classifierNode = featureNode->first_node("weakClassifiers");
		j = 0;
		for (internalStageNode = classifierNode->first_node(); internalStageNode; internalStageNode = internalStageNode->next_sibling()){
			node = internalStageNode->first_node("internalNodes");
			strtol(node->value(), &end, 10);
			strtol(end, &end, 10);
			stages[i][j].featureIdx = strtol(end, &end, 10);
			stages[i][j].threshold = strtof(end, &end);

			node = internalStageNode->first_node("leafValues");
			stages[i][j].leftWeight = strtof(node->value(), &end);
			stages[i][j].rightWeight = strtof(end, &end);
			j++;
		}

		i++;
	}

	return 0;
}

int parseFeatures(xml_node<> *root, feature_t *features, int& featureNum){
	xml_node<> *featureNode;
	xml_node<> *node;
	char *end;
	int i = 0;
	if (root == NULL){
		return -1;
	}

	for (featureNode = root->first_node(); featureNode; featureNode = featureNode->next_sibling()){
		node = featureNode->first_node("rects");
		node = node->first_node();
		features[i].black.x = strtol(node->value(), &end, 10);
		features[i].black.y = strtol(end, &end, 10);
		features[i].black.w = strtol(end, &end, 10);
		features[i].black.h = strtol(end, &end, 10);
		features[i].black.weight = strtol(end, &end, 10);

		node = node->next_sibling();
		features[i].white.x = strtol(node->value(), &end, 10);
		features[i].white.y = strtol(end, &end, 10);
		features[i].white.w = strtol(end, &end, 10);
		features[i].white.h = strtol(end, &end, 10);
		features[i].white.weight = strtol(end, &end, 10);
		i++;
	}

	return 0;
}

int countFeatures(xml_node<> *root){
	xml_node<> *featureNode;
	int i = 0;
	if (root == NULL){
		return -1;
	}

	for (featureNode = root->first_node(); featureNode; featureNode = featureNode->next_sibling()){
		i++;
	}

	return i;
}