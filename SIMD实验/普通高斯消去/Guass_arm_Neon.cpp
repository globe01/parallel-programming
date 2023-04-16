# include <iostream>
# include <arm_neon.h>
# include <sys/time.h>

using namespace std;

const int n = 250;//500,1000
float M[n][n];
float T[n][n];


//定义所需向量寄存器
float32x4_t va = vmovq_n_f32(0);
float32x4_t vx = vmovq_n_f32(0);
float32x4_t vaij = vmovq_n_f32(0);
float32x4_t vaik = vmovq_n_f32(0);
float32x4_t vakj = vmovq_n_f32(0);


//测试用例生成
void m_reset()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            M[i][j] = 0;
        }
        M[i][i] = 1.0;
        for (int j = i + 1; j < n; j++)
            M[i][j] = rand();
    }

    for (int k = 0; k < n; k++)
    {
        for (int i = k + 1; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                M[i][j] += M[k][j];
            }
        }
    }
}

//串行
void serial()
{
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];//除法步骤，整行除以第一个的系数
        }
        M[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                M[i][j] = M[i][j] - M[i][k] * M[k][j];//消去步骤
            }
            M[i][k] = 0;//左下角化为0了
        }
    }
}

//串行 cache优化
void serial_cache()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            T[j][i] = M[i][j];
            M[i][j] = 0;
        }
    }
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];
        }
        M[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                M[i][j] = M[i][j] - T[k][i] * M[k][j];
            }
        }
    }
}



//并行 未对齐
void Neon()
{
    for (int k = 0; k < n; k++)
    {
        //串行算法中二重循环的优化
        float32x4_t vt = vmovq_n_f32(M[k][k]);
        int j;
        for (j = k + 1; j + 4 <= n; j += 4)
        {
            va = vld1q_f32(&(M[k][j]));//将四个单精度浮点数从内存加载到向量寄存器
            va = vdivq_f32(va, vt);//向量对位相除
            vst1q_f32(&(M[k][j]), va);//将四个单精度浮点数从向量寄存器存储到内存
        }
        for (; j < n; j++)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];//该行结尾处有几个元素还未计算

        }
        M[k][k] = 1.0;
        //串行算法中三重循环的优化
        for (int i = k + 1; i < n; i++)
        {
            vaik = vmovq_n_f32(M[i][k]);
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                vakj = vld1q_f32(&(M[k][j]));
                vaij = vld1q_f32(&(M[i][j]));
                vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&M[i][j], vaij);
            }
            for (; j < n; j++)
            {
                M[i][j] = M[i][j] - M[i][k] * M[k][j];
            }
            M[i][k] = 0;
        }
    }
}

//并行 cache优化
void Neon_cache()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            T[j][i] = M[i][j];
            M[i][j] = 0;
        }
    }
    for (int k = 0; k < n; k++)
    {
        float32x4_t vt = vmovq_n_f32(M[k][k]);
        int j;
        for (j = k + 1; j + 4 <= n; j += 4)
        {
            va = vld1q_f32(&(M[k][j]));
            va = vdivq_f32(va, vt);
            vst1q_f32(&(M[k][j]), va);
        }

        for (; j < n; j++)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];

        }
        M[k][k] = 1.0;

        for (int i = k + 1; i < n; i++)
        {
            vaik = vmovq_n_f32(T[k][i]);

            for (j = k + 1; j + 4 <= n; j += 4)
            {
                vakj = vld1q_f32(&(M[k][j]));
                vaij = vld1q_f32(&(M[i][j]));
                vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);

                vst1q_f32(&M[i][j], vaij);
            }

            for (; j < n; j++)
            {
                M[i][j] = M[i][j] - M[i][k] * M[k][j];
            }
        }
    }
}

//并行 对齐
void Neon_Alignment()
{
    for (int k = 0; k < n; k++)
    {
        //串行算法中二重循环的优化
        float32x4_t vt = vmovq_n_f32(M[k][k]);
        int j = k + 1;
        //对齐优化，先串行处理到对齐边界
        while ((k * n + j) % 4 != 0)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];
            j++;
        }
        //其余并行处理
        for (; j + 4 <= n; j += 4)
        {
            va = vld1q_f32(&M[k][j]);//将四个单精度浮点数从内存加载到向量寄存器
            va = vdivq_f32(va, vt);//向量对位相除
            vst1q_f32(&M[k][j], va);//将四个单精度浮点数从向量寄存器存储到内存
        }
        for (; j < n; j++)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];//该行结尾处有几个元素还未计算
        }
        M[k][k] = 1.0;
        //串行算法中三重循环的优化
        for (int i = k + 1; i < n; i++)
        {
            vaik = vmovq_n_f32(M[i][k]);
            int j = k + 1;
            //对齐优化，先串行处理到对齐边界
            while ((i * n + j) % 4 != 0)
            {
                M[i][j] = M[i][j] - M[k][j] * M[i][k];
                j++;
            }
            //其余并行处理
            for (; j + 4 <= n; j += 4) 
            {
                vakj = vld1q_f32(&M[k][j]);
                vaij = vld1q_f32(&M[i][j]);
                vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&M[i][j], vaij);
            }
            for (; j < n; j++) {
                M[i][j] = M[i][j] - M[k][j] * M[i][k];
            }
            M[i][k] = 0.0;
        }
    }
}


//仅进行串行算法中二重循环的优化
void OnlyTwo()
{
    for (int k = 0; k < n; k++)
    {
        //串行算法中二重循环的优化
        float32x4_t vt = vmovq_n_f32(M[k][k]);
        int j;
        for (j = k + 1; j + 4 <= n; j += 4)
        {
            va = vld1q_f32(&(M[k][j]));
            va = vdivq_f32(va, vt);
            vst1q_f32(&(M[k][j]), va);
        }

        for (; j < n; j++)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];
        }
        M[k][k] = 1.0;

        //串行算法中三重循环不变
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                M[i][j] = M[i][j] - M[i][k] * M[k][j];
            }
            M[i][k] = 0;
        }
    }
}


//仅进行串行算法中三重循环的优化
void OnlyThree()
{
    int j;
    for (int k = 0; k < n; k++)
    {
        //串行算法中二重循环不变
        for (j = k + 1; j < n; j++)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];
        }
        M[k][k] = 1.0;
        //串行算法中三重循环的优化
        for (int i = k + 1; i < n; i++)
        {
            vaik = vmovq_n_f32(M[i][k]);
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                vakj = vld1q_f32(&(M[k][j]));
                vaij = vld1q_f32(&(M[i][j]));
                vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);

                vst1q_f32(&M[i][j], vaij);
            }
            for (; j < n; j++)
            {
                M[i][j] = M[i][j] - M[i][k] * M[k][j];
            }
            M[i][k] = 0;
        }
    }
}




int main() {
    struct timeval head, tail;
    //测量串行时间
    m_reset();
    gettimeofday(&head, NULL);
    //计时开始
    serial();
    //计时结束
    gettimeofday(&tail, NULL);
    double seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Serial: " << seconds << " ms" << endl;

    //测量串行 cache优化时间
    m_reset();
    gettimeofday(&head, NULL);
    //计时开始
    serial_cache();
    //计时结束
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "serial_cache: " << seconds << " ms" << endl;

    //测量并行 未对齐时间
    m_reset();
    gettimeofday(&head, NULL);
    //计时开始
    Neon();
    //计时结束
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Neon: " << seconds << " ms" << endl;

    //测量并行 cache优化时间
    m_reset();
    gettimeofday(&head, NULL);
    //计时开始
    Neon_cache();
    //计时结束
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Neon_cache: " << seconds << " ms" << endl;

    //测量并行 对齐时间
    m_reset();
    gettimeofday(&head, NULL);
    //计时开始
    Neon_Alignment();
    //计时结束
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Neon_Alignment: " << seconds << " ms" << endl;

    //测量仅优化二重循环的时间
    m_reset();
    gettimeofday(&head, NULL);
    //计时开始
    OnlyTwo();
    //计时结束
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "OnlyTwo: " << seconds << " ms" << endl;

    //测量仅优化三重循环的时间
    m_reset();
    gettimeofday(&head, NULL);
    //计时开始
    OnlyThree();
    //计时结束
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "OnlyThree: " << seconds << " ms" << endl;
}
