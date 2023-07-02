# include <iostream>
# include <arm_neon.h>
# include <sys/time.h>

using namespace std;

const int n = 250;//500,1000
float M[n][n];
float T[n][n];


//�������������Ĵ���
float32x4_t va = vmovq_n_f32(0);
float32x4_t vx = vmovq_n_f32(0);
float32x4_t vaij = vmovq_n_f32(0);
float32x4_t vaik = vmovq_n_f32(0);
float32x4_t vakj = vmovq_n_f32(0);


//������������
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

//����
void serial()
{
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];//�������裬���г��Ե�һ����ϵ��
        }
        M[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                M[i][j] = M[i][j] - M[i][k] * M[k][j];//��ȥ����
            }
            M[i][k] = 0;//���½ǻ�Ϊ0��
        }
    }
}

//���� cache�Ż�
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



//���� δ����
void Neon()
{
    for (int k = 0; k < n; k++)
    {
        //�����㷨�ж���ѭ�����Ż�
        float32x4_t vt = vmovq_n_f32(M[k][k]);
        int j;
        for (j = k + 1; j + 4 <= n; j += 4)
        {
            va = vld1q_f32(&(M[k][j]));//���ĸ������ȸ��������ڴ���ص������Ĵ���
            va = vdivq_f32(va, vt);//������λ���
            vst1q_f32(&(M[k][j]), va);//���ĸ������ȸ������������Ĵ����洢���ڴ�
        }
        for (; j < n; j++)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];//���н�β���м���Ԫ�ػ�δ����

        }
        M[k][k] = 1.0;
        //�����㷨������ѭ�����Ż�
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

//���� cache�Ż�
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

//���� ����
void Neon_Alignment()
{
    for (int k = 0; k < n; k++)
    {
        //�����㷨�ж���ѭ�����Ż�
        float32x4_t vt = vmovq_n_f32(M[k][k]);
        int j = k + 1;
        //�����Ż����ȴ��д�������߽�
        while ((k * n + j) % 4 != 0)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];
            j++;
        }
        //���ಢ�д���
        for (; j + 4 <= n; j += 4)
        {
            va = vld1q_f32(&M[k][j]);//���ĸ������ȸ��������ڴ���ص������Ĵ���
            va = vdivq_f32(va, vt);//������λ���
            vst1q_f32(&M[k][j], va);//���ĸ������ȸ������������Ĵ����洢���ڴ�
        }
        for (; j < n; j++)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];//���н�β���м���Ԫ�ػ�δ����
        }
        M[k][k] = 1.0;
        //�����㷨������ѭ�����Ż�
        for (int i = k + 1; i < n; i++)
        {
            vaik = vmovq_n_f32(M[i][k]);
            int j = k + 1;
            //�����Ż����ȴ��д�������߽�
            while ((i * n + j) % 4 != 0)
            {
                M[i][j] = M[i][j] - M[k][j] * M[i][k];
                j++;
            }
            //���ಢ�д���
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


//�����д����㷨�ж���ѭ�����Ż�
void OnlyTwo()
{
    for (int k = 0; k < n; k++)
    {
        //�����㷨�ж���ѭ�����Ż�
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

        //�����㷨������ѭ������
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


//�����д����㷨������ѭ�����Ż�
void OnlyThree()
{
    int j;
    for (int k = 0; k < n; k++)
    {
        //�����㷨�ж���ѭ������
        for (j = k + 1; j < n; j++)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];
        }
        M[k][k] = 1.0;
        //�����㷨������ѭ�����Ż�
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
    //��������ʱ��
    m_reset();
    gettimeofday(&head, NULL);
    //��ʱ��ʼ
    serial();
    //��ʱ����
    gettimeofday(&tail, NULL);
    double seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Serial: " << seconds << " ms" << endl;

    //�������� cache�Ż�ʱ��
    m_reset();
    gettimeofday(&head, NULL);
    //��ʱ��ʼ
    serial_cache();
    //��ʱ����
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "serial_cache: " << seconds << " ms" << endl;

    //�������� δ����ʱ��
    m_reset();
    gettimeofday(&head, NULL);
    //��ʱ��ʼ
    Neon();
    //��ʱ����
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Neon: " << seconds << " ms" << endl;

    //�������� cache�Ż�ʱ��
    m_reset();
    gettimeofday(&head, NULL);
    //��ʱ��ʼ
    Neon_cache();
    //��ʱ����
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Neon_cache: " << seconds << " ms" << endl;

    //�������� ����ʱ��
    m_reset();
    gettimeofday(&head, NULL);
    //��ʱ��ʼ
    Neon_Alignment();
    //��ʱ����
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Neon_Alignment: " << seconds << " ms" << endl;

    //�������Ż�����ѭ����ʱ��
    m_reset();
    gettimeofday(&head, NULL);
    //��ʱ��ʼ
    OnlyTwo();
    //��ʱ����
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "OnlyTwo: " << seconds << " ms" << endl;

    //�������Ż�����ѭ����ʱ��
    m_reset();
    gettimeofday(&head, NULL);
    //��ʱ��ʼ
    OnlyThree();
    //��ʱ����
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "OnlyThree: " << seconds << " ms" << endl;
}
