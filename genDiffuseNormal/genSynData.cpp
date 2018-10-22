#include "pfm.h"
#include <iostream>
#include <ctime>
#include "perlin.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;

double rd()
{
	return double(rand() * (RAND_MAX + 1.0) + rand()) / (RAND_MAX + 1.0) / (RAND_MAX + 1.0);
}

double* perlin(siv::PerlinNoise noise)
{
	noise.reseed(time(0));
	double scale = rd() * 10 + 10;
	double bx = rd() * 100000;
	double by = rd() * 100000;
	double *img = new double[256 * 256];
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			img[y * 256 + x] = noise.noise0_1((x + bx) / scale, (y + by) / scale);
		}
	}
	return img;
}

double* base()
{
	double color[3] = { rd(), rd(), rd() };
	double *img = new double[256 * 256 * 3];
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			img[(y * 256 + x) * 3] = color[0];
			img[(y * 256 + x) * 3 + 1] = color[1];
			img[(y * 256 + x) * 3 + 2] = color[2];
		}
	}
	return img;
}

double* smooth(const siv::PerlinNoise noise)
{
	double color[3] = { rd(), rd(), rd() };
	double *img = new double[256 * 256 * 3];
	double *pimg = perlin(noise);
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			img[(y * 256 + x) * 3] = color[0] * pimg[y * 256 + x];
			img[(y * 256 + x) * 3 + 1] = color[1] * pimg[y * 256 + x];
			img[(y * 256 + x) * 3 + 2] = color[2] * pimg[y * 256 + x];
		}
	}
	delete[] pimg;
	return img;
}

double* sharp(const siv::PerlinNoise noise)
{
	double color[3] = { rd(), rd(), rd() };
	double *img = new double[256 * 256 * 3];
	double *pimg = perlin(noise);
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			img[(y * 256 + x) * 3] = color[0] * max(min((pimg[y * 256 + x] - 0.499) * 500.0, 1.0), 0.0);
			img[(y * 256 + x) * 3 + 1] = color[1] * max(min((pimg[y * 256 + x] - 0.499) * 500.0, 1.0), 0.0);
			img[(y * 256 + x) * 3 + 2] = color[2] * max(min((pimg[y * 256 + x] - 0.499) * 500.0, 1.0), 0.0);
		}
	}
	delete[] pimg;
	return img;
}

void imgAdd(double* a, double* b)
{
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			a[(y * 256 + x) * 3] += b[(y * 256 + x) * 3];
			a[(y * 256 + x) * 3 + 1] += b[(y * 256 + x) * 3 + 1];
			a[(y * 256 + x) * 3 + 2] += b[(y * 256 + x) * 3 + 2];
		}
	}
}

void hAdd(double* a, double* b)
{
	for (int y = 0; y < 256; y++)
	{
		for (int x = 0; x < 256; x++)
		{
			a[y * 256 + x] += b[y * 256 + x];
		}
	}
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
	cv::Mat matNm(256, 256, CV_32FC3, normal);
	cv::GaussianBlur(matNm, matNm, cv::Size(5, 5), 0.0);
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


double* genAlbedo(int te, int sh, const siv::PerlinNoise noise)
{
	double* al = base();
	for (int i = 0; i < te; i++)
	{
		double* img = smooth(noise);
		imgAdd(al, img);
		delete[] img;
	}
	for (int i = 0; i < sh; i++)
	{
		double* img = sharp(noise);
		imgAdd(al, img);
		delete[] img;
	}

	return al;
}

bool comp(pair<double, int> a, pair<double, int> b)
{
	return a.first < b.first;
}

int main(int argc, char** argv)
{
	if (argc != 4) {
		cout << "Invalid input! Please input: texture pattern mode, number of data in thousand, output folder" << endl;
		cout << "texture pattern mode: 1 for regular, 2 for smooth only, 3 for sharp only" << endl;
		cout << "number of data in thousand: eg. 5 for 5000 pairs of data (albedo and normal), 10 for 10000 pairs of data" << endl;
		return -1;
	}
	int mode = atoi(argv[1]); //input 1 for regular, 2 for smooth only, 3 for sharp only
	int iters = atoi(argv[2]); //number of data to generated, counted in thousand, eg. input 5 means generating 5000 data
	char* root = argv[3]; //output folder

	siv::PerlinNoise noise(time(0));
	srand(time(0));
	auto *alarray = new pair<double, int>*[3];
	for (int i = 0; i < 3; i++)
	{
		alarray[i] = new pair<double, int>[1000 * 256 * 256];
	}

	for (int iter = 0; iter < iters; iter++)
	{

		//#pragma omp parallel for
		for (int i = 0; i < 1000; i++)
		{
			double* ret;
			
			if (mode == 1) {
				if (i < 200)
				{
					ret = genAlbedo(1, 0, noise);
				}
				else if (i < 400)
				{
					ret = genAlbedo(0, 1, noise);
				}
				else if (i < 600)
				{
					ret = genAlbedo(2, 0, noise);
				}
				else if (i < 800)
				{
					ret = genAlbedo(0, 2, noise);
				}
				else
				{
					ret = genAlbedo(1, 1, noise);
				}
			}
			else if (mode == 2)
			{
				if (i < 500)
				{
					ret = genAlbedo(1, 0, noise);
				}
				else
				{
					ret = genAlbedo(2, 0, noise);
				}
			}
			else
			{
				if (i < 500)
				{
					ret = genAlbedo(0, 1, noise);
				}
				else
				{
					ret = genAlbedo(0, 2, noise);
				}
			}

			for (int j = 0; j < 256 * 256; j++)
			{
				alarray[0][i * 256 * 256 + j] = { ret[j * 3], i * 256 * 256 + j };
				alarray[1][i * 256 * 256 + j] = { ret[j * 3 + 1], i * 256 * 256 + j };
				alarray[2][i * 256 * 256 + j] = { ret[j * 3 + 2], i * 256 * 256 + j };
			}

			delete[] ret;
		}

#pragma omp parallel for
		for (int c = 0; c < 3; c++)
		{
			sort(alarray[c], alarray[c] + 1000 * 256 * 256, comp);
			for (int i = 0; i < 1000 * 256 * 256; i++)
			{
				alarray[c][alarray[c][i].second].first = i / (1000 * 256 * 256.0);
			}
		}

#pragma omp parallel for
		for (int i = 0; i < 1000; i++)
		{
			HDRImage img;
			img.setSize(256, 256);
			img.data = new float[256 * 256 * 3];
			double* al = new double[256 * 256 * 3];
			for (int j = 0; j < 256 * 256; j++)
			{
				img.data[j * 3] = alarray[0][i * 256 * 256 + j].first;
				img.data[j * 3 + 1] = alarray[1][i * 256 * 256 + j].first;
				img.data[j * 3 + 2] = alarray[2][i * 256 * 256 + j].first;
				al[j * 3] = alarray[0][i * 256 * 256 + j].first;
				al[j * 3 + 1] = alarray[1][i * 256 * 256 + j].first;
				al[j * 3 + 2] = alarray[2][i * 256 * 256 + j].first;
			}
			char buffer[100];
			snprintf(buffer, sizeof(buffer), "%s/%d_albedo.pfm", root, iter * 1000 + i);
			img.writePfm(buffer);
			delete[] img.data;

			img.data = height2normal(albedo2height(al));
			snprintf(buffer, sizeof(buffer), "%s/%d_normal.pfm", root, iter * 1000 + i);
			img.writePfm(buffer);
			delete[] img.data;
			delete[] al;
		}
	}

	return 0;
}