// LINEAR_REGRESSION.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstring>
#include <fstream>
#include <ctime>

using namespace std;

#define SHRESHOLD 0.1
#define ALPHA 0.05

int main()
{
	static double x[25000][384];
	double theta[384] = { 1 };
	double newtheta[384] = { 1 };
	double reference[25000];
	double Jtheta = 99999999999;
	bool isConverge = false;

	//	read the train set
	FILE *fptrain;
	fptrain = fopen("C:\\Users\\LIU\\Desktop\\dataMining\\sample_submission.txt", "r");
	if (!fptrain) {
		cout << "file C:\\Users\\LIU\\Desktop\\dataMining\\sample_submission.txt open fail." << endl;
		system("pause");
		return 0;
	}

	double start = clock();
	double finish;

	int id;
	for (int i = 0; i < 282795; i++) {
		fscanf(fptrain, "%d,", &id);
		for (int j = 0; j < 384; j++) {
			fscanf(fptrain, "%lf,", &x[i][j]);
		}
		fscanf(fptrain, "%lf", &reference[i]);
	}

	/*the following code can be used to check whether the IO is correct
	printf("%lf, %lf\n", x[10485][383], reference[10485]);
	printf("%lf, %lf\n", x[24999][383], reference[24999]);*/

	fclose(fptrain);  
	finish = clock();
	printf("%lf seconds\n", (finish - start) / CLOCKS_PER_SEC);

	//	read the theta
	FILE *fptry;
	fptry = fopen("C:\\Users\\LIU\\Desktop\\try.csv", "r");
	if (!fptry) {
		cout << "file C:\\Users\\LIU\\Desktop\\try.csv open fail." << endl;
		system("pause");
		return 0;
	}
	for (int i = 0; i < 383; i++) {
		fscanf(fptry, "%lf,", &theta[i]);
	}
	fscanf(fptry, "%lf", &theta[383]);
	fclose(fptry);

	int round = 0;
	while (!isConverge) {
		cout << round++ << "round" << endl;
		double h[25000] = { 0 };
		double difference[25000] = { 0 };

		//	compute htheta and htheta - y
		for (int i = 0; i < 25000; i++) {
			for (int j = 0; j < 384; j++) {
				h[i] += theta[j] * x[i][j];
			}
			difference[i] = h[i] - reference[i];
		}
		
		//	compute the Jtheta
		double newJtheta = 0;
		for (int i = 0; i < 20000; i++) {
			newJtheta += (h[i] - reference[i]) * (h[i] - reference[i]);
		}

		//	the second method to judge convergence
		/*if (-SHRESHOLD < newJtheta - Jtheta && newJtheta - Jtheta < SHRESHOLD) {
			isConverge = true;
		}*/
		Jtheta = newJtheta;
		printf("%lf\n", Jtheta);

		//	compute the sum in the formula and new theta
		double sum[384] = { 0 };
		for (int i = 0; i < 384; i++) {
			for (int j = 0; j < 25000; j++) {
				sum[i] += difference[j] * x[j][i];
			}
			newtheta[i] = theta[i] - ALPHA * sum[i] / 25000;
		}

		//	the first method to judge convergence
		/*isConverge = true;
		for (int i = 0; i < 384; i++) {
			if (newtheta[i] - theta[i] < -SHRESHOLD || newtheta[i] - theta[i] > SHRESHOLD) {
				isConverge = false;
				break;
			}
		}*/

		FILE *fp;
		fp = fopen("C:\\Users\\LIU\\Desktop\\sample_submission.txt", "r");
		if (!fp) {
			cout << "file C:\\Users\\LIU\\Desktop\\dataMining\\sample_submission.txt open fail." << endl;
			system("pause");
			return 0;
		}
		//	update the old theta with the new one
		for (int i = 0; i < 384; i++) {
			theta[i] = newtheta[i];
			fprintf(fp, "%lf,", theta[i]);
		}
		fclose(fp);

		FILE *fp;
		fp = fopen("C:\\Users\\LIU\\Desktop\\sample_submission.txt", "w");
		if (!fp) {
			cout << "file C:\\Users\\LIU\\Desktop\\dataMining\\sample_submission.txt open fail." << endl;
			system("pause");
			return 0;
		}
		//	update the old theta with the new one
		for (int i = 0; i < 384; i++) {
			theta[i] = newtheta[i];
			fprintf(fp, "%lf,", theta[i]);
		}
		fclose(fp);
	}
	system("pause");
    return 0;
}


