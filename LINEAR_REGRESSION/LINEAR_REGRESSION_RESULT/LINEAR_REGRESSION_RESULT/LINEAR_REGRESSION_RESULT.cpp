// LINEAR_REGRESSION_RESULT.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <cstdlib>

using namespace std;


int main()
{
	double theta[384] = { 0 };
	static double train[25000][384] = {0};
	double h[25000] = {0};

	//	read the theta
	FILE *fptry;
	fptry = fopen("C:\\Users\\LIU\\Desktop\\try.csv", "r");
	if (!fptry) {
		cout << "file C:\\Users\\LIU\\Desktop\\dataMining\\save_train.csv open fail." << endl;
		system("pause");
		return 0;
	}
	for (int i = 0; i < 383; i++) {
		fscanf(fptry, "%lf,", &theta[i]);
	}
	fscanf(fptry, "%lf", &theta[383]);
	fclose(fptry);

	//	read the train set
	FILE *fptest;
	fptest = fopen("C:\\Users\\LIU\\Desktop\\dataMining\\save_test.csv", "r");
	if (!fptest) {
		cout << "file C:\\Users\\LIU\\Desktop\\dataMining\\save_test.csv open fail." << endl;
		system("pause");
		return 0;
	}
	int id;
	char a[5000];
	for (int i = 0; i < 25000; i++) {
		fscanf(fptest, "%d,", &id);
		for (int j = 0; j < 383; j++) {
			fscanf(fptest, "%lf,", &train[i][j]);
		}
		fscanf(fptest, "%lf", &train[i][383]);
	}
	fclose(fptest);

	//	compute the prediction
	for (int i = 0; i < 25000; i++) {
		for (int j = 0; j < 384; j++) {
			h[i] += theta[j] * train[i][j];
		}
	}

	//	write to the output file
	FILE *fpoutput;
	fpoutput = fopen("C:\\Users\\LIU\\Desktop\\dataMining\\output.csv", "w");
	if (!fpoutput) {
		cout << "file C:\\Users\\LIU\\Desktop\\dataMining\\output.csv open fail." << endl;
		system("pause");
		return 0;
	}
	id = 0;
	fprintf(fpoutput, "Id,reference\n");
	for (int i = 0; i < 25000; i++) {
		fprintf(fpoutput, "%d,%lf\n", id, h[i]);
		id++;
	}
	fclose(fpoutput);

    return 0;
}

