#include <omp.h>
#include <iostream>
#include <windows.h>
using namespace std;

const int N = 1000;



float tmp_M[N][N];
float M[N][N];
const int NUM_THREADS = 5; //线程数

void m_reset()
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			tmp_M[i][j] = 0;
		}
		tmp_M[i][i] = 1.0;
		for (int j = i + 1; j < N; j++)
			tmp_M[i][j] = rand() % 100;
	}
	for (int k = 0; k < N; k++)
	{
		for (int i = k + 1; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				tmp_M[i][j] += tmp_M[k][j];
			}
		}
	}
}

void newM()
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			M[i][j] = tmp_M[i][j];
	}
}


void Serial()
{
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];
		}
		M[k][k] = 1.0;

		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0.0;
		}
	}
}

void Static()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < N; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = M[k][k];
			for (int j = k + 1; j < N; j++)
			{
				M[k][j] = M[k][j] / tmp;
			}
			M[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(Static)
		for (int i = k + 1; i < N; i++)
		{
			float tmp = M[i][k];
			for (int j = k + 1; j < N; j++)
				M[i][j] = M[i][j] - tmp * M[k][j];
			M[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}


void Dynamic()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < N; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = M[k][k];
			for (int j = k + 1; j < N; j++)
			{
				M[k][j] = M[k][j] / tmp;
			}
			M[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(dynamic, 80)
		for (int i = k + 1; i < N; i++)
		{
			float tmp = M[i][k];
			for (int j = k + 1; j < N; j++)
				M[i][j] = M[i][j] - tmp * M[k][j];
			M[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}


int main()
{
	m_reset();
	double seconds;
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	newM();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	Serial();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "Serial: " << seconds << "ms" << endl;


	newM();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	Static();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "Static: " << seconds << "ms" << endl;


	newM();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	Dynamic();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "Dynamic: " << seconds << "ms" << endl;
	return 0;
}