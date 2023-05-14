#include <iostream>
#include <arm_neon.h>
#include <sys/time.h>
#include <pthread.h>

using namespace std;

const int N = 1000;//问题规模500,1000,1500
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

//串行
void serial()
{
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];//除法步骤，整行除以第一个的系数
		}
		M[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];//消去步骤
			}
			M[i][k] = 0;//左下角化为0了
		}
	}
}


struct threadParam_t
{
	int k; //消去的轮次
	int t_id; // 线程 id
};
int worker_count = 5; //工作线程数量
void* threadFunc(void* param)
{
	//定义所需向量寄存器
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);

	threadParam_t* p = (threadParam_t*)param;
	int k = p->k; //消去的轮次
	int t_id = p->t_id; //线程编号
	int i = k + t_id + 1; //获取自己的计算任务
	for (int m = k + 1 + t_id; m < N; m += worker_count)
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
			M[m][j] = M[m][j] - M[m][k] * M[k][j];
		}
		M[m][k] = 0;
	}
	pthread_exit(NULL);
	return 0;
}


int main()
{
	double seconds;//总时间
	struct timeval head, tail;
	m_reset();
	gettimeofday(&head, NULL);


	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vt = vmovq_n_f32(0);

	for (int k = 0; k < N; k++)
	{
		//串行算法中二重循环的优化
		vt = vmovq_n_f32(M[k][k]);
		int j;
		for (j = k + 1; j + 4 <= N; j += 4)
		{
			va = vld1q_f32(&(M[k][j]));//将四个单精度浮点数从内存加载到向量寄存器
			va = vdivq_f32(va, vt);//向量对位相除
			vst1q_f32(&(M[k][j]), va);//将四个单精度浮点数从向量寄存器存储到内存
		}

		for (; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];//该行结尾处有几个元素还未计算

		}
		M[k][k] = 1.0;

		//串行算法中三重循环的优化
		//各行之间互不影响，可采用多线程执行
		//创建工作线程，进行消去操作
		pthread_t* handles = new pthread_t[worker_count];// 创建对应的 Handle
		threadParam_t* param = new threadParam_t[worker_count];// 创建对应的线程数据结构

		//分配任务
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//创建线程
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);

		//主线程挂起等待所有的工作线程完成此轮消去工作
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);

	}
	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "pthread-dynamic-neon: " << seconds << "ms" << endl;
	return 0;
}
