#include <omp.h>
#include <iostream>
#include <sys/time.h>
# include <arm_neon.h>
using namespace std;

const int N = 1500;



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
			M[i][k] = 0,0;
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
#pragma omp for schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			float tmp = M[i][k];
			for (int j = k + 1; j < N; j++)
				M[i][j] = M[i][j] - tmp * M[k][j];
			M[i][k] = 0.0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}
void Static_simd()
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
#pragma omp for simd
		for (int i = k + 1; i < N; i++)
		{
			int tmp = M[i][k];
			for (int j = k + 1; j < N; j++)
				M[i][j] = M[i][j] - tmp * M[k][j];
			M[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}


//静态 neon barrier
void Static_neon_barrier()
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);

#pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < N; k++)
	{
		//串行部分
#pragma omp master
		{
			float32x4_t vt = vmovq_n_f32(M[k][k]);
			int j;
			for (j = k + 1; j < N; j++)
			{
				va = vld1q_f32(&(M[k][j]));
				va = vdivq_f32(va, vt);
				vst1q_f32(&(M[k][j]), va);
			}
			for (; j < N; j++)
			{
				M[k][j] = M[k][j] * 1.0 / M[k][k];

			}
			M[k][k] = 1.0;
		}

		//并行部分
#pragma omp barrier
#pragma omp for schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			vaik = vmovq_n_f32(M[i][k]);
			int j;
			for (j = k + 1; j + 4 <= N; j += 4)
			{
				vakj = vld1q_f32(&(M[k][j]));
				vaij = vld1q_f32(&(M[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);

				vst1q_f32(&M[i][j], vaij);
			}

			for (; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}

			M[i][k] = 0;
		}
	}
}


//每次循环均重新创建线程
void Static_create_dy()
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);

	for (int k = 0; k < N; k++)
	{
		//串行部分
		{
			float32x4_t vt = vmovq_n_f32(M[k][k]);
			int j;
			for (j = k + 1; j < N; j++)
			{
				va = vld1q_f32(&(M[k][j]));
				va = vdivq_f32(va, vt);
				vst1q_f32(&(M[k][j]), va);
			}
			for (; j < N; j++)
			{
				M[k][j] = M[k][j] * 1.0 / M[k][k];

			}
			M[k][k] = 1.0;
		}

		//并行部分
#pragma omp parallel for num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj) ,schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			vaik = vmovq_n_f32(M[i][k]);
			int j;
			for (j = k + 1; j + 4 <= N; j += 4)
			{
				vakj = vld1q_f32(&(M[k][j]));
				vaij = vld1q_f32(&(M[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);

				vst1q_f32(&M[i][j], vaij);
			}

			for (; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}

			M[i][k] = 0;
		}
	}
}




//静态 neon 二重循环的除法部分也并行化
void Static_neon_division()
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);

#pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < N; k++)
	{
		//除法部分
#pragma omp for schedule(static)
		for (int j = k + 1; j < N; j++)
		{
			M[k][j] = M[k][j] / M[k][k];
		}
		M[k][k] = 1.0;

		//并行部分
#pragma omp for schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			vaik = vmovq_n_f32(M[i][k]);
			int j;
			for (j = k + 1; j + 4 <= N; j += 4)
			{
				vakj = vld1q_f32(&(M[k][j]));
				vaij = vld1q_f32(&(M[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);

				vst1q_f32(&M[i][j], vaij);
			}

			for (; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}

			M[i][k] = 0;
		}
	}
}





int main()
{
	m_reset();
	struct timeval head, tail;
	double seconds;

	newM();
	gettimeofday(&head, NULL);
	Serial();
	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "Serial: " << seconds << "ms" << endl;


	newM();
	gettimeofday(&head, NULL);
	Static();
	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "Static: " << seconds << "ms" << endl;




	newM();
	gettimeofday(&head, NULL);
	Static_simd();
	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "Static_simd: " << seconds << "ms" << endl;


	newM();
	gettimeofday(&head, NULL);
	Static_neon_barrier();
	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "Static_neon_barrier: " << seconds << "ms" << endl;
	
	newM();
	gettimeofday(&head, NULL);
	Static_create_dy();
	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "Static_create_dy: " << seconds << "ms" << endl;



	newM();
	gettimeofday(&head, NULL);
	Static_neon_division();
	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "Static_neon_division: " << seconds << "ms" << endl;



	return 0;
}