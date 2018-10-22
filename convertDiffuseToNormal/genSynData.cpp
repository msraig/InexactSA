#include "pfm.h"
#include <iostream>
#include <ctime>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace std;
using namespace cv;

double rd()
{
	return double(rand() * (RAND_MAX + 1.0) + rand()) / (RAND_MAX + 1.0) / (RAND_MAX + 1.0);
}

double* albedo2height(double* albedo, double sharpness = 0.1)
{
	double *gray = new double[256 * 256];
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			gray[y * 256 + x] = albedo[(y * 256 + x) * 3] * 0.299 + albedo[(y * 256 + x) * 3 + 1] * 0.587 + albedo[(y * 256 + x) * 3 + 2] * 0.114;
		}
	}
	double gmax = *max_element(gray, gray + 256 * 256);
	double gmin = *min_element(gray, gray + 256 * 256);
	double scale = sharpness * 512.0 * (rd() - 0.5) / max(gmax - gmin, 0.05);
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			gray[y * 256 + x] *= scale;
		}
	}
	return gray;
}

float* height2normal(double* height)
{
	float *normal = new float[256 * 256 * 3];
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			double nm[3] = { 0.0,0.0,0.0 };
			if (y > 0 && x > 0) {
				double a = height[(y - 1) * 256 + x] - height[y * 256 + x];
				double b = height[y * 256 + x - 1] - height[y * 256 + x];
				double len = sqrt(1.0 + a*a + b*b);
				nm[0] += b / len;
				nm[1] += -a / len;
				nm[2] += 1 / len;
			}
			if (y < 255 && x > 0) {
				double a = height[y * 256 + x - 1] - height[y * 256 + x];
				double b = height[(y + 1) * 256 + x] - height[y * 256 + x];
				double len = sqrt(1.0 + a*a + b*b);
				nm[0] += a / len;
				nm[1] += b / len;
				nm[2] += 1 / len;
			}
			if (y < 255 && x < 255) {
				double a = height[(y + 1) * 256 + x] - height[y * 256 + x];
				double b = height[y * 256 + x + 1] - height[y * 256 + x];
				double len = sqrt(1.0 + a*a + b*b);
				nm[0] += -b / len;
				nm[1] += a / len;
				nm[2] += 1 / len;
			}
			if (y > 0 && x < 255) {
				double a = height[y * 256 + x + 1] - height[y * 256 + x];
				double b = height[(y - 1) * 256 + x] - height[y * 256 + x];
				double len = sqrt(1.0 + a*a + b*b);
				nm[0] += -a / len;
				nm[1] += -b / len;
				nm[2] += 1 / len;
			}
			double len = sqrt(nm[0] * nm[0] + nm[1] * nm[1] + nm[2] * nm[2]);
			normal[(y * 256 + x) * 3] = nm[0] / len;
			normal[(y * 256 + x) * 3 + 1] = nm[1] / len;
			normal[(y * 256 + x) * 3 + 2] = nm[2] / len;
		}
	}
	Mat matNm(256, 256, CV_32FC3, normal);
	GaussianBlur(matNm, matNm, Size(5, 5), 0.0);
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			double len = sqrt(normal[(y * 256 + x) * 3] * normal[(y * 256 + x) * 3] + normal[(y * 256 + x) * 3 + 1] * normal[(y * 256 + x) * 3 + 1] + normal[(y * 256 + x) * 3 + 2] * normal[(y * 256 + x) * 3 + 2]);
			normal[(y * 256 + x) * 3] = (normal[(y * 256 + x) * 3] / len + 1) / 2;
			normal[(y * 256 + x) * 3 + 1] = (normal[(y * 256 + x) * 3 + 1] / len + 1) / 2;
			normal[(y * 256 + x) * 3 + 2] = (normal[(y * 256 + x) * 3 + 2] / len + 1) / 2;
		}
	}
	for (int i = 0; i < 256 * 256 * 3; i++) {
		if (normal[i] < 0.0) {
			cout << "<0!!!" << endl;
			normal[i] = 0.0;
		}
		else if (normal[i] > 1.0) {
			cout << ">1!!!" << endl;
			normal[i] = 1.0;
		}
	}
	delete[] height;
	return normal;
}


int main(int argc, char** argv)
{
	if (argc != 4) {
		cout << "Invalid input! Please input: number of images, input folder, output folder" << endl;
		cout << "The input images must be named as \'0.png\' to \'n-1.png\'" << endl;
		return -1;
	}	
	int filecnt = atoi(argv[1]);
	char* rootin = argv[2]; //input folder
	char* rootout = argv[3]; //output folder

	srand(time(0));
	for (int i = 0; i < filecnt; i++)
	{
		char buffer[100];
		snprintf(buffer, sizeof(buffer), "%s/%d.png", rootin, i);
		Mat matimg = imread(buffer);
		resize(matimg, matimg, Size(256, 256));
		double* albedo = new double[256 * 256 * 3];
		for (int y = 0; y < 256; y++)
		{
			for (int x = 0; x < 256; x++)
			{
				albedo[(y * 256 + x) * 3] = pow(matimg.at<uchar>((y * 256 + x) * 3 + 2) / 255.0, 2.2);
				albedo[(y * 256 + x) * 3 + 1] = pow(matimg.at<uchar>((y * 256 + x) * 3 + 1) / 255.0, 2.2);
				albedo[(y * 256 + x) * 3 + 2] = pow(matimg.at<uchar>((y * 256 + x) * 3) / 255.0, 2.2);
			}
		}
		float* normal = height2normal(albedo2height(albedo));

		HDRImage img;
		img.setSize(256, 256);
		img.data = new float[256 * 256 * 3];
		for (int j = 0; j < 256 * 256 * 3; j++)
		{
			img.data[j] = albedo[j];
		}
		snprintf(buffer, sizeof(buffer), "%s/%d_albedo.pfm", rootout, i);
		img.writePfm(buffer);
		delete[] img.data;
		img.data = normal;
		snprintf(buffer, sizeof(buffer), "%s/%d_normal.pfm", rootout, i);
		img.writePfm(buffer);
		delete[] albedo;
		delete[] normal;
	}
	
	return 0;
}