// RANDOM_FOREST.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <utility>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <ctime>
#include <thread>
#include <mutex>

#define NUMBER_OF_TRAIN_SAMPLES 1000
#define NUMBER_OF_TEST_SAMPLES 50000
#define NUMBER_OF_TREES 1866
#define NUMBER_OF_THREAD 4

using std::vector;
using std::map;
using std::string;
using std::endl;
using std::cout;
using std::make_pair;
using std::pair;

using namespace std;

string g_stTrainPath = "C:\\Users\\LIU\\Desktop\\classification\\train_data.txt";
string g_stTestPath = "C:\\Users\\LIU\\Desktop\\classification\\test_data.txt";
string g_stPredictionPath = "C:\\Users\\LIU\\Desktop\\classification\\preds.txt";

//	打开测试文件
static int prediction[NUMBER_OF_TEST_SAMPLES][NUMBER_OF_TREES];

struct CSample {
	int m_nLabel;
	map<int, double> m_mFeaturesAndValues;
};

vector<CSample> testSet;


class CNode {
public:
	CNode() {}
	CNode(vector<CSample> samples) {
		init(samples);
		//__CalculateGini();
		m_bIsLeaf = true;
	}
	void init(vector<CSample> samples) {
		m_vSamples = samples;
		m_nSplitFeature = -1;
		m_dSplitPoint = -1;
		m_dEigenvalue = -1;
		m_nLeft = -2;
		m_nRight = -2;
		m_nJudge = -3;
		//specialKind = false;
		m_nNumberOfSamples = samples.size();
		for (int i = 0; i < m_nNumberOfSamples; i++) {
			map<int, double>::iterator begin = samples[i].m_mFeaturesAndValues.begin();
			map<int, double>::iterator end = samples[i].m_mFeaturesAndValues.end();
			for (map<int, double>::iterator iter = begin; iter != end; iter++) {
				m_vAllFeatures.insert(iter->first);
			}
			//cout << "The " << i << "th data has " << m_vSamples[i].m_vFeatures.size() << " features" << endl;
		}
		m_nNumberOfSelectedFeatures = (int)sqrt(m_vAllFeatures.size());

		/*cout << "Total number of the feature:" << m_vAllFeatures.size() << endl;
		cout << "So we are to pick " << sqrt(m_vAllFeatures.size()) << " features." << endl;
		cout << "The rear feature is " << *(m_vAllFeatures.rbegin()) << endl;
		for (set<int>::iterator iter = m_vAllFeatures.begin(); iter != m_vAllFeatures.end(); iter++) {
		cout << *iter << " ";
		}
		cout << endl;*/
	}
	void setIsleaf(bool isleaf) {
		m_bIsLeaf = isleaf;
	}
	bool getIsleaf() {
		return m_bIsLeaf;
	}
	double getGini() {
		return m_dGini;
	}
	int getSplitFeature() {
		return m_nSplitFeature;
	}
	double getSplitPoint() {
		return m_dSplitPoint;
	}
	double getEigenvalue() {
		return m_dEigenvalue;
	}
	int getLeft() {
		return m_nLeft;
	}
	void setLeft(int l) {
		m_nLeft = l;
	}
	int getRight() {
		return m_nRight;
	}
	void setRight(int r) {
		m_nRight = r;
	}
	int getJudge() {
		return m_nJudge;
	}
	void calculateJudge() {
		int class0 = 0;
		int class1 = 0;
		for (int i = 0; i < m_nNumberOfSamples; i++) {
			if (m_vSamples[i].m_nLabel == 0) class0++;
			else class1++;
		}
		if (class0 <= class1) m_nJudge = 1;
		else m_nJudge = 0;
	}
	void setJudge(int label) {
		m_nJudge = label;
	}
	vector<CSample> getSamples() {
		return m_vSamples;
	}
	double CalculateGini() {
		int class1 = 0;
		int class2 = 0;
		for (int i = 0; i < m_nNumberOfSamples; i++) {
			if (m_vSamples[i].m_nLabel == 0) class1++;
			else class2++;
		}
		m_dGini = 1 - (double(class1) / m_vSamples.size()) * (double(class1) / m_vSamples.size())
			- (double(class2) / m_vSamples.size()) * (double(class2) / m_vSamples.size());
		return m_dGini;
		//cout << m_dGini << endl;
	}
	void SelectMFeatures() {
		set<int> tempSet;
		while (tempSet.size() != m_nNumberOfSelectedFeatures) {
			int temp = rand() % 202;
			if (m_vAllFeatures.count(temp) != 0) tempSet.insert(temp);
		}
		m_vSelectedFeatures = tempSet;
	}
	void Split() {
		//	装某一特征的所有值
		set<double> currentValues;
		map<double, vector<int>> labelsOfCurrentValues;
		double minSumOfGiniIndex = 2;

		//	对所有特征的所有分裂点进行基尼系数的计算，选出最好的特征和分裂点并返回两个子节点
		for (auto feature : m_vSelectedFeatures) {
			//	currentValues装下一个特征的所有值之前清空上一个特征的所有值
			currentValues.clear();
			labelsOfCurrentValues.clear();
			//cout << feature << ": ";
			
			int count = 0;
			double sum = 0;
			vector<int> defaultLabels;
			//	对拥有该feature的样本，该feature对应的值加入集合currentValues，而该样本加入samplesOfCurrentValues
			for (auto sample : m_vSamples) {
				map<int, double>::iterator tempIter = sample.m_mFeaturesAndValues.find(feature);
				//	该样本中有找到该feature
				if (tempIter != sample.m_mFeaturesAndValues.end()) {
					currentValues.insert(tempIter->second);
					labelsOfCurrentValues[tempIter->second].push_back(sample.m_nLabel);
					//	用VS查找debug时要用所选内容来查找否则文件太大很难从头开始
					//cout << sample.m_mFeaturesAndValues[*tempIter] << " ";
					sum += tempIter->second;
					count++;
				}
				//	该样本中没有找到该feature
				else {
					defaultLabels.push_back(sample.m_nLabel);
				}
			}

			//	计算该特征的均值并赋值给缺省值
			double average = sum / count;
			//cout << endl << "The average value of the feature " << feature <<  " is " << average << endl;
			currentValues.insert(average);
			labelsOfCurrentValues[average].insert(labelsOfCurrentValues[average].end(), defaultLabels.begin(), defaultLabels.end());

			//	初始化左右容器
			double tempSplitPoint = 0;
			//cout << feature << " split points: ";
			vector<int> leftVector;
			vector<int> rightVector;
			for (set<double>::iterator iter = currentValues.begin(); iter != currentValues.end(); iter++) {
				rightVector.insert(rightVector.end(), labelsOfCurrentValues[*iter].begin(), labelsOfCurrentValues[*iter].end());
			}

			if (currentValues.size() == 1) continue;
			//	对该feature计算所有分裂点，并从该分裂点列出两个子节点
			for (set<double>::iterator iter = currentValues.begin(); iter != currentValues.end(); iter++) {
				set<double>::iterator nextIter = ++iter;
				iter--;
				if (nextIter != currentValues.end()) {
					tempSplitPoint = ((*iter) + (*nextIter)) / 2.0f;
					//cout << tempSplitPoint << " ";
					//	根据分裂点分开两类样本

					leftVector.insert(leftVector.end(), labelsOfCurrentValues[*iter].begin(), labelsOfCurrentValues[*iter].end());
					rightVector.erase(rightVector.begin(), rightVector.begin() + labelsOfCurrentValues[*iter].size());

					//	计算左子基尼系数
					int leftClass1 = 0;
					int leftClass2 = 0;
					double leftGini;
					for (int i = 0; i < leftVector.size(); i++) {
						if (leftVector[i] == 0) leftClass1++;
						else leftClass2++;
					}
					leftGini = 1 - (double(leftClass1) / leftVector.size()) * (double(leftClass1) / leftVector.size())
						- (double(leftClass2) / leftVector.size()) * (double(leftClass2) / leftVector.size());

					//	计算右子基尼系数
					int rightClass1 = 0;
					int rightClass2 = 0;
					double rightGini;
					for (int i = 0; i < rightVector.size(); i++) {
						if (rightVector[i] == 0) rightClass1++;
						else rightClass2++;
					}
					rightGini = 1 - (double(rightClass1) / rightVector.size()) * (double(rightClass1) / rightVector.size())
						- (double(rightClass2) / rightVector.size()) * (double(rightClass2) / rightVector.size());

					//	加权后基尼系数
					double gini = leftGini * ((double)leftVector.size() / (leftVector.size() + rightVector.size())) +
						rightGini * ((double)rightVector.size() / (leftVector.size() + rightVector.size()));

					//cout << tempSplitPoint <<  ": " << gini << endl;
					if (gini < minSumOfGiniIndex) {
						m_nSplitFeature = feature;
						m_dSplitPoint = tempSplitPoint;
						minSumOfGiniIndex = gini;
						m_dEigenvalue = average;
					}
				}
				//	没有分裂点了
				else {
					break;
				}
			}
			/*cout << endl << "The best split feature is " << m_nSplitFeature << " and the best split point is "<< m_dSplitPoint <<
				" and the minSumOfGiniIndex is " << minSumOfGiniIndex <<endl;*/
			if (m_nSplitFeature == -1) {
				//	这种情况是currentValues中2个数据太过接近以至无法比较区分
				calculateJudge();
				/*cout << "Size of samples:" << m_vSamples.size() << endl;
				cout << "Average:" << average << " currentValues.size():" << currentValues.size() << endl;
				auto testiter = currentValues.begin();
				cout << "Values are:" << *testiter;
				testiter++;
				cout << " " << *testiter << endl;
				cout << "Feature is:" << feature << endl;
				cout << "tempSplitPoint:" << tempSplitPoint << " leftVector.size():" << leftVector.size() <<" rightVector.size():" << rightVector.size() << endl;
				system("pause");*/
			}
		}

		m_bIsLeaf = false;
	}
private:
	//void __CalculateEntropy() {
	//	int class1 = 0;
	//	int class2 = 0;
	//	for (int i = 0; i < m_vSamples.size(); i++) {
	//		if (m_vSamples[i].m_nLabel == 0) class1++;
	//		else class2++;
	//	}
	//	//cout << "The number of class1:" << class1 << endl;
	//	//cout << "The number of class2:" << class2 << endl;
	//	double proportionOfSample1 = (double)class1 / m_vSamples.size();
	//	double proportionOfSample2 = (double)class2 / m_vSamples.size();
	//	if (proportionOfSample1 == 0 || proportionOfSample2 == 0) {
	//		m_dEntropy = 0;
	//	}
	//	else {
	//		m_dEntropy = -proportionOfSample1 * log(proportionOfSample1) - proportionOfSample2 * log(proportionOfSample2);
	//	}
	//	//cout << "The entropy of " << m_vSamples.size() << " class:" << m_dEntropy << endl;
	//}

	int m_nNumberOfSamples;
	vector<CSample> m_vSamples;
	set<int> m_vAllFeatures;
	int m_nNumberOfSelectedFeatures;
	set<int> m_vSelectedFeatures;
	//double m_dEntropy;
	double m_dGini;
	int m_nSplitFeature;
	double m_dSplitPoint;
	double m_dEigenvalue;
	bool m_bIsLeaf;
	int m_nLeft;
	int m_nRight;
	int m_nJudge;
};

struct CPredictNode
{
public:
	CPredictNode(CNode n) {
		m_nSplitFeature = n.getSplitFeature();
		m_dSplitPoint = n.getSplitPoint();
		m_dEigenvalue = n.getEigenvalue();
		m_nLeft = n.getLeft();
		m_nRight = n.getRight();
		m_nJudge = n.getJudge();
	}
public:
	int m_nSplitFeature;
	double m_dSplitPoint;
	double m_dEigenvalue;
	int m_nLeft;
	int m_nRight;
	int m_nJudge;
};

class CDecisionTree {
public:
	CDecisionTree(vector<CSample> samples) {
		//	构建根节点
		CNode root(samples);
		root.CalculateGini();
		//	cout << "Root gini:" << root.getGini() << endl;
		m_vNodes.push_back(root);
		__Train();
		__Record();
	}
	~CDecisionTree() {}
	int pred(CSample sample) {
		int label = -1;
		CPredictNode node = m_vPNodes[0];
		while (true)
		{
			if (node.m_nLeft == -2) {
				return node.m_nJudge;
			}
			else {
				auto splitfeature = node.m_nSplitFeature;
				auto splitpoint = node.m_dSplitPoint;
				if (sample.m_mFeaturesAndValues.find(splitfeature) != sample.m_mFeaturesAndValues.end()) {
					if (sample.m_mFeaturesAndValues[splitfeature] <= splitpoint) {
						node = m_vPNodes[node.m_nLeft];
					}
					else {
						node = m_vPNodes[node.m_nRight];
					}
				}
				else {
					auto average = node.m_dEigenvalue;
					if (average <= splitpoint) {
						node = m_vPNodes[node.m_nLeft];
					}
					else {
						node = m_vPNodes[node.m_nRight];
					}
				}
			}
		}

		cout << "Out of the for loop" << endl;
		system("pause");
		return label;
	}
private:
	void __Train() {
		vector<CSample> left;
		vector<CSample> right;
		while (left.empty() && right.empty())
		{
			vector<CNode>::iterator iter;
			for (iter = m_vNodes.begin(); iter != m_vNodes.end(); iter++) {
				//	非叶子节点或者叶子结点中只有一类
				if (iter->getIsleaf() == false || iter->getGini() == 0) {
					continue;
				}
				srand(unsigned(time(NULL)));
				iter->SelectMFeatures();
				iter->Split();

				if (iter->getSplitFeature() != -1) {
					auto splitfeature = iter->getSplitFeature();
					auto splitpoint = iter->getSplitPoint();
					auto oldSamples = iter->getSamples();
					auto averageEigenvalue = iter->getEigenvalue();
					for (auto sample : oldSamples) {
						if (sample.m_mFeaturesAndValues.find(splitfeature) != sample.m_mFeaturesAndValues.end()) {
							if (sample.m_mFeaturesAndValues[splitfeature] <= splitpoint) {
								left.push_back(sample);
							}
							else {
								right.push_back(sample);
							}
						}
						else {
							if (averageEigenvalue <= splitpoint) {
								left.push_back(sample);
							}
							else {
								right.push_back(sample);
							}
						}
					}
					break;
				}
				else {
					//	这种情况是缺失值赋为均值后导致存在样本无法区分
					iter->calculateJudge();
					/*cout << endl << "No split feature and split point was found" << endl;
					cout << "The size of samples is:" << iter->getSamples().size() << endl;*/
				}
			}

			if (!left.empty()) {
				CNode l(left);
				CNode r(right);
				if (l.CalculateGini() == 0) {
					l.setJudge((*(l.getSamples().begin())).m_nLabel);
				}
				if (r.CalculateGini() == 0) {
					r.setJudge((*(r.getSamples().begin())).m_nLabel);
				}
				/*cout << endl << "Node1 gini:" << l.getGini() << endl;
				cout << "Node2 gini:" << r.getGini() << endl;*/
				iter->setLeft(m_vNodes.size());
				iter->setRight(m_vNodes.size() + 1);
				m_vNodes.push_back(l);
				m_vNodes.push_back(r);
				left.clear();
				right.clear();
			}
			else {
				//	cout << endl << "Congragulations!The tree is completed!" << endl;
				break;
			}
		}
	}
	void __Record() {
		for (vector<CNode>::iterator iter = m_vNodes.begin(); iter != m_vNodes.end(); iter++) {
			CPredictNode temp(*iter);
			m_vPNodes.push_back(temp);
		}
		m_vNodes.clear();
	}
	vector<CNode> m_vNodes;
	vector<CPredictNode> m_vPNodes;
};

mutex mtx;

void thread1(int start)
{
	for (int r = start; r < start + NUMBER_OF_TREES / NUMBER_OF_THREAD; r++) {
		//mtx.lock();

		//	打开训练文件
		FILE *fptrain = fopen(g_stTrainPath.c_str(), "r");
		if (!fptrain) {
			cout << "file " + g_stTrainPath + " open fail." << endl;
			return;
		}

		double start = clock();
		srand(time(NULL));
		//	读文件
		vector<CSample> samples;
		for (int i = 0; i < NUMBER_OF_TRAIN_SAMPLES; i++) {
			auto offset = rand() % 748995920;
			rewind(fptrain);
			fseek(fptrain, offset, 0);
			while (fgetc(fptrain) != '\n')
			{
				char s[100];
				fscanf(fptrain, "%s", s);
			}

			CSample newSample;
			fscanf(fptrain, "%d", &newSample.m_nLabel);
			int newFeature = 0;
			double newValue = 0;
			char temp = fgetc(fptrain);
			while (temp != '\n') {
				fscanf(fptrain, "%d:%lf", &newFeature, &newValue);
				temp = fgetc(fptrain);
				newSample.m_mFeaturesAndValues[newFeature] = newValue;
			}
			samples.push_back(newSample);
		}

		//	关闭文件
		fclose(fptrain);
		//mtx.unlock();

		//构造决策树
		CDecisionTree decisiontree(samples);

		for (int i = 0; i < NUMBER_OF_TEST_SAMPLES; i++) {
			int label = decisiontree.pred(testSet[i]);
			prediction[i][r] = label;
		}
	}
}

int main()
{
	FILE *fptest = fopen(g_stTrainPath.c_str(), "r");
	if (!fptest) {
		cout << "file " + g_stTrainPath + " open fail." << endl;
		return 0;
	}

	//	读测试文件
	for (int i = 0; i < NUMBER_OF_TEST_SAMPLES; i++) {
		CSample newSample;
		int id = 0;
		fscanf(fptest, "%d", &id);
		int newFeature = 0;
		double newValue = 0;
		char temp = fgetc(fptest);
		while (temp != '\n') {
			fscanf(fptest, "%d:%lf", &newFeature, &newValue);
			temp = fgetc(fptest);
			newSample.m_mFeaturesAndValues[newFeature] = newValue;
		}
		testSet.push_back(newSample);
	}

	double start = clock();

	thread* threads = new thread[NUMBER_OF_THREAD];
	for (int i = 0; i < NUMBER_OF_THREAD; i++) {
		threads[i] = thread(thread1, NUMBER_OF_TREES / NUMBER_OF_THREAD * i);
	}
	for (int i = 0; i < NUMBER_OF_THREAD; i++) {
		threads[i].join();
	}
	delete[] threads;

	double forestComplete = clock();
	printf("It took %d threads %lfs to construct %d trees and predicts.\n", NUMBER_OF_THREAD, (forestComplete - start) / CLOCKS_PER_SEC,
		NUMBER_OF_TREES);

	//	打开输出文件
	FILE *fppred = fopen(g_stPredictionPath.c_str(), "w");
	if (!fppred) {
		cout << "file " + g_stPredictionPath + " open fail." << endl;
		return 0;
	}
	fprintf(fppred, "id,label\n");

	//	写入输出文件
	for (int i = 0; i < NUMBER_OF_TEST_SAMPLES; i++) {
		int sum = 0;
		for (int j = 0; j < NUMBER_OF_TREES; j++) {
			sum += prediction[i][j];
		}
		fprintf(fppred, "%d,%lf\n", i, (double)sum / NUMBER_OF_TREES);
	}

	double finish = clock();

	printf("It took %d threads %lfs to construct %d trees, predict and write to the file.\n", NUMBER_OF_THREAD, (finish-start)/CLOCKS_PER_SEC, 
		NUMBER_OF_TREES);
	
	fclose(fptest);
	fclose(fppred);

	system("pause");
	return 0;
}








