#include <stdio.h>
#include <iostream>
#include "RapidXML\rapidxml.hpp"
#include "RapidXML\rapidxml_utils.hpp"
#include "types.h"
#include "xmlparse.h"


using namespace std;
using namespace rapidxml;

int parseStages(xml_node<> *root, stage_t **stages, stageMeta_t *stagesMeta);
int countStages(xml_node<> *root);
void writeStagesMeta(xml_node<> *root, stageMeta_t *&stagesMeta, int stagenum);

int parseClassifier(char *filename, int& stageNum, stageMeta_t*& stagesMeta, stage_t**& stages, feature_t*& features){
	int i = 0;

	file<> xmlfile(filename); // default template is char
	xml_document<> doc;
	doc.parse<0>(xmlfile.data());

	xml_node<> *node = doc.first_node()->first_node()->first_node("stages");

	stageNum = countStages(node);
	stagesMeta = new stageMeta_t[stageNum]; // not arbitrary. fix later?
	stages = new stage_t*[stageNum];

	writeStagesMeta(node, stagesMeta, stageNum);

	parseStages(node, stages, stagesMeta);
	
	return 0;
}

int parseStages(xml_node<> *root, stage_t **stages, stageMeta_t*stagesMeta){
	xml_node<> *stageNode;
	xml_node<> *treeNode;
	xml_node<> *featureNode;
	xml_node<> *node;
	xml_node<> *rectNode;
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

	for (stageNode = root->first_node(); stageNode; stageNode = stageNode->next_sibling()){
		stages[i] = new stage_t[stagesMeta[i].size];

		j = 0;
		treeNode = stageNode->first_node("trees");
		for (internalStageNode = treeNode->first_node(); internalStageNode; internalStageNode = internalStageNode->next_sibling()){

			featureNode = internalStageNode->first_node();

			node = featureNode->first_node("threshold");
			stages[i][j].threshold = strtof(node->value(), &end);

			node = featureNode->first_node("left_val");
			stages[i][j].leftWeight = strtof(node->value(), &end);

			node = featureNode->first_node("right_val");
			stages[i][j].rightWeight = strtof(node->value(), &end);

			rectNode = featureNode->first_node("feature")->first_node("rects");

			node = rectNode->first_node();
			stages[i][j].feature.black.x = strtol(node->value(), &end, 10);
			stages[i][j].feature.black.y = strtol(end, &end, 10);
			stages[i][j].feature.black.w = strtol(end, &end, 10);
			stages[i][j].feature.black.h = strtol(end, &end, 10);
			stages[i][j].feature.black.weight = strtol(end, &end, 10);

			node = node->next_sibling();
			stages[i][j].feature.white.x = strtol(node->value(), &end, 10);
			stages[i][j].feature.white.y = strtol(end, &end, 10);
			stages[i][j].feature.white.w = strtol(end, &end, 10);
			stages[i][j].feature.white.h = strtol(end, &end, 10);
			stages[i][j].feature.white.weight = strtol(end, &end, 10);

			node = node->next_sibling();
			if (node){
				stages[i][j].feature.third.x = strtol(node->value(), &end, 10);
				stages[i][j].feature.third.y = strtol(end, &end, 10);
				stages[i][j].feature.third.w = strtol(end, &end, 10);
				stages[i][j].feature.third.h = strtol(end, &end, 10);
				stages[i][j].feature.third.weight = strtol(end, &end, 10);
			}
			else{
				stages[i][j].feature.third.x = 0;
				stages[i][j].feature.third.y = 0;
				stages[i][j].feature.third.w = 0;
				stages[i][j].feature.third.h = 0;
				stages[i][j].feature.third.weight = 0;
			}

			j++;
		}

		i++;
	}

	return 0;
}

int countStages(xml_node<> *root){
	xml_node<> *featureNode;
	xml_node<> *treeNode;
	xml_node<> *stageNode;
	int i = 0;
	if (root == NULL){
		return -1;
	}

	for (featureNode = root->first_node(); featureNode; featureNode = featureNode->next_sibling()){
		i++;
	}

	return i;
}

void writeStagesMeta(xml_node<> *root, stageMeta_t *&stagesMeta, int stagenum){
	xml_node<> *featureNode;
	xml_node<> *treeNode;
	xml_node<> *stageNode;
	int i = 0;
	int count = 0;
	if (root == NULL){
		return;
	}

	for (featureNode = root->first_node(); featureNode; featureNode = featureNode->next_sibling()){
		treeNode = featureNode->first_node();
		count = 0;
		for (stageNode = treeNode->first_node(); stageNode; stageNode = stageNode->next_sibling()){
			count++;
		}
		stagesMeta[i].size = count;
		treeNode = featureNode->first_node("stage_threshold");
		stagesMeta[i].threshold = strtof(treeNode->value(), NULL);
		i++;
	}

}