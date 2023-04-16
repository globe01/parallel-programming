#include <iostream>
#include <sstream>
#include <fstream>
#include <arm_neon.h>
#include <sys/time.h>

using namespace std;


//����9
unsigned int Core[37960][1188] = { 0 };//����Ԫ���Ƿ�Ϊ0
unsigned int Rows[37960][1188] = { 0 };//����Ԫ��
const int Rows_num = 14291;//����Ԫ����
const int N = 1187;
const int column = 37960;//��������
const int each_col = 10;//ÿ�ִ�����Ԫ�ӵ�����


//����8
//unsigned int Core[23075][722] = { 0 };//����Ԫ���Ƿ�Ϊ0
//unsigned int Rows[23075][722] = { 0 };//����Ԫ��
//const int Rows_num = 14325;//����Ԫ����
//const int N = 721;
//const int column = 23075;//��������
//const int each_col = 10;//ÿ�ִ�����Ԫ�ӵ�����


////��ȡ��Ԫ�Ӳ���ʼ��
void Core_reset()
{
    unsigned int a;
    ifstream infile("Core_9.txt");
    char fin[100000] = { 0 };
    int curr;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int p = 0;
        while (line >> a)
        {
            if (p == 0)
            {
                curr = a;
                p = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            Core[curr][N-1 - j] += temp;
            Core[curr][N] = 1;//1�ǿ���0��
        }
    }
}


//��ȡ����Ԫ�в���ʼ��
void Rows_reset()
{
    unsigned int a;
    ifstream infile("Rows_9.txt");
    char fin[100000] = { 0 };
    int curr = 0;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int p = 0;
        while (line >> a)
        {
            if (p == 0)
            {
                Rows[curr][N] = a;
                p = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            Rows[curr][N-1 - j] += temp;
        }
        curr++;
    }
}


//�����㷨
void Serial()
{
    for (int i = column - 1; i - each_col >= -1; i -= each_col)//������Ԫ�ӵ��У�ÿ�ֵ���each_col����Ԫ��
    {
        for (int j = 0; j < Rows_num; j++)//��������Ԫ�У��ҵ���0Ԫ��
        {
            while (Rows[j][N] <= i && Rows[j][N] >= i - each_col+1)//����0Ԫ���ڵ�ǰ��Ԫ�Ӵ���Χ��
            {
                int curr = Rows[j][N];
                if (Core[curr][N] == 1)//��Ԫ�ӷǿ�ʱ������ȥ����������֮�����µ��׸���0Ԫ��
                {
                    for (int k = 0; k < N; k++)
                    {
                        Rows[j][k] = Rows[j][k] ^ Core[curr][k];//��һѭ�����ֺ����ɽ���SIMD�Ż�
                    }
                    int t = 0;
                    for (int m = 0; m < N; m++)
                    {
                        if (Rows[j][m] != 0)
                        {
                            unsigned int temp = Rows[j][m];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                t++;
                            }
                            t += m * 32;
                            break;
                        }
                    }
                    Rows[j][N] = t - 1;

                }
                else//��Ԫ��Ϊ�գ�����Ԫ������Ϊ��Ԫ��
                {
                    for (int k = 0; k < N; k++)
                    {
                        Core[curr][k] = Rows[j][k];
                    }
                    Core[curr][N] = 1;
                    break;
                }
            }
        }
    }
}


//Neon
void Neon()
{
    for (int i = column - 1; i - each_col >= -1; i -= each_col)
    {
        for (int j = 0; j < Rows_num; j++)
        {
            while (Rows[j][N] <= i && Rows[j][N] >= i - each_col+1)
            {
                int curr = Rows[j][N];
                if (Core[curr][N] == 1)
                {
                    //������ѭ������Neon�Ż�
                    int k;
                    for (k = 0; k + each_col/2 <= N; k += each_col/2)
                    {
                        uint32x4_t vzi = vld1q_u32(&(Core[curr][k]));//��Ԫ�Ӽ��ص������Ĵ���
                        uint32x4_t vhang = vld1q_u32(&(Rows[j][k]));//����Ԫ�м��ص������Ĵ���
                        vhang = veorq_u32(vhang, vzi);//�������
                        vst1q_u32(&(Rows[j][k]), vhang);//�洢���ڴ�
                    }
                    for (; k < N; k++)
                    {
                        Rows[j][k] = Rows[j][k] ^ Core[curr][k];//��β���м�����δ����
                    }
                    int m = 0, t = 0;
                    for (m = 0; m < N; m++)
                    {
                        if (Rows[j][m] != 0)
                        {
                            unsigned int temp = Rows[j][m];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                t++;
                            }
                            t += m * 32;
                            break;
                        }
                    }
                    Rows[j][N] = t - 1;
                }
                else
                {
                    for (int k = 0; k < N; k++)
                    {
                        Core[curr][k] = Rows[j][k];
                    }
                    Core[curr][N] = 1;
                    break;
                }
            }
        }
    }
}


int main(){
    struct timeval head, tail;
    
    //����
    Core_reset();
    Rows_reset();
    gettimeofday(&head, NULL);
    Serial();
    gettimeofday(&tail, NULL);
    double seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Serial: " << seconds << " ms" << endl;

    //Neon
    Core_reset();
    Rows_reset();
    gettimeofday(&head, NULL);
    Neon();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Neon: " << seconds << " ms" << endl;

}