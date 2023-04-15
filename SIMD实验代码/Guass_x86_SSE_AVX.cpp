#include <iostream>
#include <windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2、AVX-512
using namespace std;

const int N = 1000;//问题规模
float M[N][N];

//测试用例生成
void m_reset()
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			M[i][j] = 0;
		}
		M[i][i] = 1.0;
		for (int j = i + 1; j < N; j++)
			M[i][j] = rand();
	}
	for (int k = 0; k < N; k++)
	{
		for (int i = k + 1; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				M[i][j] += M[k][j];
			}
		}
	}
}

//串行算法
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
			M[i][k] = 0;
		}
	}
}

//SSE优化 未对齐
__m128 va, vt, vx, vaij, vaik, vakj;
void SSE()
{
	for (int k = 0; k < N; k++)
	{
		//串行算法中二重循环的优化
		vt = _mm_set_ps(M[k][k], M[k][k], M[k][k], M[k][k]);
		int j;
		for (j = k + 1; j + 4 <= N; j += 4)
		{
			va = _mm_loadu_ps(&(M[k][j]));//将四个单精度浮点数从内存加载到向量寄存器
			va = _mm_div_ps(va, vt);//向量对位相除
			_mm_store_ps(&(M[k][j]), va);//将四个单精度浮点数从向量寄存器存储到内存
		}
		for (; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];//该行结尾处有几个元素还未计算
		}
		M[k][k] = 1.0;
		//串行算法中三重循环的优化
		for (int i = k + 1; i < N; i++)
		{
			vaik = _mm_set_ps(M[i][k], M[i][k], M[i][k], M[i][k]);
			for (j = k + 1; j + 4 <= N; j += 4)
			{
				vakj = _mm_loadu_ps(&(M[k][j]));
				vaij = _mm_loadu_ps(&(M[i][j]));
				vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_store_ps(&M[i][j], vaij);
			}
			for (; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
	}
}


//SSE优化 对齐
void SSE_Alignment() {
	for (int k = 0; k < N; k++) {
		//串行算法中二重循环的优化
		vt = _mm_set_ps(M[k][k], M[k][k], M[k][k], M[k][k]);
		int j = k + 1;
		//对齐优化
		while ((k * N + j) % 4 != 0) {
			M[k][j] = M[k][j] * 1.0 / M[k][k];
			j++;	
		}
		for (; j + 4 <= N; j += 4) {
			va = _mm_load_ps(&(M[k][j]));//将四个单精度浮点数从内存加载到向量寄存器
			va = _mm_div_ps(va, vt);//向量对位相除
			_mm_store_ps(&(M[k][j]), va);//将四个单精度浮点数从向量寄存器存储到内存
		}
		for (; j < N; j++) {
			M[k][j] = M[k][j] * 1.0 / M[k][k];//该行结尾处有几个元素还未计算
		}
		M[k][k] = 1.0;
		//串行算法中三重循环的优化
		for (int i = k + 1; i < N; i++) {
			vaik = _mm_set_ps(M[i][k], M[i][k], M[i][k], M[i][k]);
			int j = k + 1;
			//对齐优化
			while ((i * N + j) % 4 != 0) {
				M[i][j] = M[i][j] - M[k][j] * M[i][k];
				j++;
			}
			for (; j + 4 <= N; j += 4) {
				vakj = _mm_load_ps(&(M[k][j]));
				vaij = _mm_load_ps(&(M[i][j]));
				vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_store_ps(&M[i][j], vaij);
			}
			for (; j < N; j++) {
				M[i][j] = M[i][j] - M[k][j] * M[i][k];
			}
			M[i][k] = 0.0;
		}
	}
}

//SSE 仅进行串行算法中二重循环的优化
void SSE_OnlyTwo()
{
	for (int k = 0; k < N; k++)
	{
		//串行算法中二重循环的优化
		vt = _mm_set_ps(M[k][k], M[k][k], M[k][k], M[k][k]);
		int j;
		for (j = k + 1; j + 4 <= N; j += 4)
		{
			va = _mm_loadu_ps(&(M[k][j]));//将四个单精度浮点数从内存加载到向量寄存器
			va = _mm_div_ps(va, vt);//向量对位相除
			_mm_store_ps(&(M[k][j]), va);//将四个单精度浮点数从向量寄存器存储到内存
		}
		for (; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];//该行结尾处有几个元素还未计算
		}
		M[k][k] = 1.0;
		//串行算法中三重循环不变
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
	}
}

//SSE 仅进行串行算法中三重循环的优化
void SSE_OnlyThree()
{
	int j;
	for (int k = 0; k < N; k++)
	{
		//串行算法中二重循环不变
		for (int j = k + 1; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];
		}
		M[k][k] = 1.0;
		//串行算法中三重循环的优化
		for (int i = k + 1; i < N; i++)
		{
			vaik = _mm_set_ps(M[i][k], M[i][k], M[i][k], M[i][k]);
			for (j = k + 1; j + 4 <= N; j += 4)
			{
				vakj = _mm_loadu_ps(&(M[k][j]));
				vaij = _mm_loadu_ps(&(M[i][j]));
				vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_store_ps(&M[i][j], vaij);
			}
			for (; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
	}
}

//AVX优化
__m256 AVXva, AVXvt, AVXvx, AVXvaij, AVXvaik, AVXvakj;
void AVX()
{
	for (int k = 0; k < N; k++)
	{
		//串行算法中二重循环的优化
		AVXvt = _mm256_set_ps(M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k]);
		int j;
		for (j = k + 1; j + 8 <= N; j += 8)
		{
			AVXva = _mm256_loadu_ps(&(M[k][j]));//将8个单精度浮点数从内存加载到向量寄存器
			AVXva = _mm256_div_ps(AVXva, AVXvt);//向量对位相除
			_mm256_store_ps(&(M[k][j]), AVXva);//将8个单精度浮点数从向量寄存器存储到内存
		}
		for (; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];//该行结尾处有几个元素还未计算

		}
		M[k][k] = 1.0;
		//串行算法中三重循环的优化
		for (int i = k + 1; i < N; i++)
		{
			AVXvaik = _mm256_set_ps(M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k]);

			for (j = k + 1; j + 8 <= N; j += 8)
			{
				AVXvakj = _mm256_loadu_ps(&(M[k][j]));
				AVXvaij = _mm256_loadu_ps(&(M[i][j]));
				AVXvx = _mm256_mul_ps(AVXvakj, AVXvaik);
				AVXvaij = _mm256_sub_ps(AVXvaij, AVXvx);

				_mm256_store_ps(&M[i][j], AVXvaij);
			}

			for (; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}

			M[i][k] = 0;
		}
	}
}

//AVX 仅进行串行算法中二重循环的优化
void AVX_OnlyTwo()
{
	for (int k = 0; k < N; k++)
	{
		//串行算法中二重循环的优化
		AVXvt = _mm256_set_ps(M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k]);
		int j;
		for (j = k + 1; j + 8 <= N; j += 8)
		{
			AVXva = _mm256_loadu_ps(&(M[k][j]));//将8个单精度浮点数从内存加载到向量寄存器
			AVXva = _mm256_div_ps(AVXva, AVXvt);//向量对位相除
			_mm256_store_ps(&(M[k][j]), AVXva);//将8个单精度浮点数从向量寄存器存储到内存
		}
		for (; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];//该行结尾处有几个元素还未计算

		}
		M[k][k] = 1.0;
		//串行算法中三重循环不变
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
	}
}

//AVX 仅进行串行算法中三重循环的优化
void AVX_OnlyThree()
{
	int j;
	for (int k = 0; k < N; k++)
	{
		//串行算法中二重循环不变
		for (int j = k + 1; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];
		}
		M[k][k] = 1.0;
		//串行算法中三重循环的优化
		for (int i = k + 1; i < N; i++)
		{
			AVXvaik = _mm256_set_ps(M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k]);

			for (j = k + 1; j + 8 <= N; j += 8)
			{
				AVXvakj = _mm256_loadu_ps(&(M[k][j]));
				AVXvaij = _mm256_loadu_ps(&(M[i][j]));
				AVXvx = _mm256_mul_ps(AVXvakj, AVXvaik);
				AVXvaij = _mm256_sub_ps(AVXvaij, AVXvx);

				_mm256_store_ps(&M[i][j], AVXvaij);
			}

			for (; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}

			M[i][k] = 0;
		}
	}
}

//AVX-512优化
__m512 va512, vt512, vx512, vaij512, vaik512, vakj512;
void AVX_512()
{
	for (int k = 0; k < N; k++)
	{
		//串行算法中二重循环的优化
		vt512 = _mm512_set_ps(M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k]);
		int j;
		for (j = k + 1; j + 16 <= N; j += 16)
		{
			va512 = _mm512_loadu_ps(&(M[k][j]));//将16个单精度浮点数从内存加载到向量寄存器
			va512 = _mm512_div_ps(va512, vt512);//向量对位相除
			_mm512_store_ps(&(M[k][j]), va512);//将16个单精度浮点数从向量寄存器存储到内存
		}
		for (; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];//该行结尾处有几个元素还未计算

		}
		M[k][k] = 1.0;
		//串行算法中三重循环的优化
		for (int i = k + 1; i < N; i++)
		{
			vaik512 = _mm512_set_ps(M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k]);

			for (j = k + 1; j + 16 <= N; j += 16)
			{
				vakj512 = _mm512_loadu_ps(&(M[k][j]));
				vaij512 = _mm512_loadu_ps(&(M[i][j]));
				vx512 = _mm512_mul_ps(vakj512, vaik512);
				vaij512 = _mm512_sub_ps(vaij512, vx512);

				_mm512_store_ps(&M[i][j], vaij512);
			}
			for (; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
	}
}

//AVX-512 仅进行串行算法中二重循环的优化
void AVX_512_OnlyTwo()
{
	for (int k = 0; k < N; k++)
	{
		//串行算法中二重循环的优化
		vt512 = _mm512_set_ps(M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k]);
		int j;
		for (j = k + 1; j + 16 <= N; j += 16)
		{
			va512 = _mm512_loadu_ps(&(M[k][j]));//将16个单精度浮点数从内存加载到向量寄存器
			va512 = _mm512_div_ps(va512, vt512);//向量对位相除
			_mm512_store_ps(&(M[k][j]), va512);//将16个单精度浮点数从向量寄存器存储到内存
		}
		for (; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];//该行结尾处有几个元素还未计算

		}
		M[k][k] = 1.0;
		//串行算法中三重循环不变
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
	}
}

//AVX-512 仅进行串行算法中三重循环的优化
void AVX_512_OnlyThree()
{
	int j;
	for (int k = 0; k < N; k++)
	{
		//串行算法中二重循环不变
		for (int j = k + 1; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];
		}
		M[k][k] = 1.0;
		//串行算法中三重循环的优化
		for (int i = k + 1; i < N; i++)
		{
			vaik512 = _mm512_set_ps(M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k]);

			for (j = k + 1; j + 16 <= N; j += 16)
			{
				vakj512 = _mm512_loadu_ps(&(M[k][j]));
				vaij512 = _mm512_loadu_ps(&(M[i][j]));
				vx512 = _mm512_mul_ps(vakj512, vaik512);
				vaij512 = _mm512_sub_ps(vaij512, vx512);

				_mm512_store_ps(&M[i][j], vaij512);
			}
			for (; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0;
		}
	}
}


int main(){
	double seconds;//总时间
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	
	//测量串行时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//计时开始
	Serial();
	//计时结束
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "Serial：" << seconds << " ms" << endl;

	//测量SSE（不对齐）时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//计时开始
	SSE();
	//计时结束
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "SSE：" << seconds << " ms" << endl;

	//测量SSE（对齐）时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//计时开始
	SSE_Alignment();
	//计时结束
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "SSE_Alignment：" << seconds << " ms" << endl;

	//测量SSE（仅优化二重循环）时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//计时开始
	SSE_OnlyTwo();
	//计时结束
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "SSE_OnlyTwo：" << seconds << " ms" << endl;

	//测量SSE（仅优化三重循环）时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//计时开始
	SSE_OnlyThree();
	//计时结束
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "SSE_OnlyThree：" << seconds << " ms" << endl;

	//测量AVX时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//计时开始
	AVX();
	//计时结束
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "AVX：" << seconds << " ms" << endl;

	//测量AVX（仅优化二重循环）时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//计时开始
	AVX_OnlyTwo();
	//计时结束
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "AVX_OnlyTwo：" << seconds << " ms" << endl;

	//测量AVX（仅优化三重循环）时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//计时开始
	AVX_OnlyThree();
	//计时结束
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "AVX_OnlyThree：" << seconds << " ms" << endl;

	//测量AVX512时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//计时开始
	AVX_512();
	//计时结束
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "AVX-512：" << seconds << " ms" << endl;

	//测量AVX-512（仅优化二重循环）时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//计时开始
	AVX_512_OnlyTwo();
	//计时结束
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "AVX_512_OnlyTwo：" << seconds << " ms" << endl;

	//测量AVX（仅优化三重循环）时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//计时开始
	AVX_512_OnlyThree();
	//计时结束
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "AVX_512_OnlyThree：" << seconds << " ms" << endl;
	
	return 0;
}