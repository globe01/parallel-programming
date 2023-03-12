//�����������˻�

#include <iostream>
#include <windows.h>
using namespace std;
const int N = 10000;//�����ģ
double b[N][N], a[N], sum[N];

void init(int N) {
    for (int i = 0; i < N; i++) {//��ʼ����������a[i]=i
        a[i] = i;
    }
    for (int i = 0; i < N; i++) {//��ʼ����������b[i][j]=i+j
        for (int j = 0; j < N; j++) {
            b[i][j] = i + j;
        }
    }
}

void ordinary(int N) {//ƽ���㷨 ���з��ʣ�һ�����ѭ�����ڴ�ѭ��һ������ִ�У������һ���ڻ����
    for (int i = 0; i < N; i++) {
        sum[i] = 0.0;
        for (int j = 0; j < N; j++){
            sum[i] += b[j][i] * a[j];
        }
    }
}

void cache(int N) {//�Ż��㷨 ��Ϊ���з��ʾ���Ԫ�أ�һ�����ѭ�����㲻���κ�һ���ڻ���ֻ����ÿ���ڻ��ۼ�һ���˷����
    for (int i = 0; i < N; i++) {
        sum[i] = 0.0;
    }
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            sum[i] += b[j][i] * a[j];
        }
    }
}


int main()
{
    int n, step = 100;
    long counter;//�ظ�����
    double seconds;//��ʱ��
    long long head, tail, freq, here;
    init(N);
    for (n = 0; n <= 10000; n += step){
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);//��ʼ��ʱ
        counter = 0;
        while (true)
        {
            QueryPerformanceCounter((LARGE_INTEGER*)&here);//��¼��ǰʱ�䣬�����С�Ͷ��ظ�����
            if ((here - head) / freq * 1000.0 > 10) {
                break;
            }
            ordinary(n);
            //cache(n);
            counter++;
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);//������ʱ
        seconds = (tail - head) / freq *1000.0;//��λת����ms
        cout << n << ' ' << counter << ' ' << seconds << ' ' << seconds / counter << endl;
        if (n == 1000)
            step = 1000;
    }
    return 0;
}