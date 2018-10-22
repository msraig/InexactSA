#include <fstream>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <vector>

void swapChannal(float* input, int n)
{
	for (int i = 0; i<n; i++)
		std::swap(input[3 * n], input[3 * n + 2]);
}


float* loadPFM(std::string fn, int& width, int& height, int& channal)
{
	//alloc memory in this function
	std::fstream file(fn.c_str(), std::ios::in | std::ios::binary);
	std::string bands;
	float scalef, fvalue;

	file >> bands;
	file >> width;
	file >> height;
	file >> scalef;

	char c = file.get();
	if (c == '\r')
		c = file.get();
	if (c != '\n')
		return NULL;

	float* output = NULL;
	if (bands == "Pf")
	{
		channal = 1;
		output = (float*)malloc(height * width * sizeof(float));
		//reverse row
		for (int y = 0; y<height; y++)
		{
			int y_0 = height - 1 - y;
			char* ptr = (char*)&(output[y_0*width]);
			file.read(ptr, sizeof(float) * width);
		}
		file.close();
	}
	else
	{
		channal = 3;
		output = (float*)malloc(height * width * 3 * sizeof(float));
		//reverse row
		for (int y = 0; y<height; y++)
		{
			int y_0 = height - 1 - y;
			char* ptr = (char*)&(output[3*y_0*width]);
			file.read(ptr, sizeof(float) * width * 3);
		}
		file.close();
//		swapChannal(output, width * height);
	}

	return output;
}



float* loadCubeMap(std::string fn, int envWidth)
{
	//alloc memory in this function
	int height, width, ch;
	float* pfm_data = loadPFM(fn, width, height, ch);

	int faceRes = envWidth;
	float* output = (float*)malloc(6 * envWidth * envWidth * 4 * sizeof(float));

	int start_y[6];
	int start_x[6];
	start_y[0] = faceRes;
	start_y[1] = faceRes;
	start_y[2] = 0;
	start_y[3] = 2 * faceRes;
	start_y[4] = faceRes;
	start_y[5] = faceRes;

	start_x[0] = 2 * faceRes;
	start_x[1] = 0;
	start_x[2] = faceRes;
	start_x[3] = faceRes;
	start_x[4] = faceRes;
	start_x[5] = 3 * faceRes;

	if (ch == 1)
	{
		for (int f = 0; f<6; f++)
		{
			int faceStart = f*(envWidth*envWidth * 4);
			for (int y = 0; y<envWidth; y++)
			{
				for (int x = 0; x<envWidth; x++)
				{
					output[faceStart + 4 * (y*envWidth + x)] = pfm_data[(start_y[f] + y)*width + start_x[f] + x];
					output[faceStart + 4 * (y*envWidth + x) + 1] = pfm_data[(start_y[f] + y)*width + start_x[f] + x];
					output[faceStart + 4 * (y*envWidth + x) + 2] = pfm_data[(start_y[f] + y)*width + start_x[f] + x];
					output[faceStart + 4 * (y*envWidth + x) + 3] = 1.0f;
				}
			}
		}
	}
	else
	{
		for (int f = 0; f<6; f++)
		{
			int faceStart = f*(envWidth*envWidth * 4);
			for (int y = 0; y<envWidth; y++)
			{
				for (int x = 0; x<envWidth; x++)
				{
					output[faceStart + 4 * (y*envWidth + x)] = pfm_data[3 * ((start_y[f] + y)*width + start_x[f] + x) + 2];
					output[faceStart + 4 * (y*envWidth + x) + 1] = pfm_data[3 * ((start_y[f] + y)*width + start_x[f] + x) + 1];
					output[faceStart + 4 * (y*envWidth + x) + 2] = pfm_data[3 * ((start_y[f] + y)*width + start_x[f] + x)];
					output[faceStart + 4 * (y*envWidth + x) + 3] = 1.0f;
				}
			}
		}
	}

	free(pfm_data);
	return output;
}
