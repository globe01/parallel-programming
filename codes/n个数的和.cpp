//n�������
#include <iostream>
#include <windows.h>
using namespace std;
const long long int N = 1048576;//2��20�η�
long long int sum;
double a[N];

void init(int n){//��ʼ��
    for (long long int i = 0; i < N; i++){
        a[i] = i;
    }
}

void ordinary(int n){//ƽ���㷨
    for (int i = 0; i < n; i++){
        sum += a[i];
    }
}

void multi(int n){//��·��ʽ
    long long int sum1 = 0, sum2 = 0;
    for (long long int i = 0; i < n; i += 2){//����Ϊ2
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;
}

void recursion(int n){//�ݹ麯�����ŵ��Ǽ򵥣�ȱ���ǵݹ麯�����ÿ����ϴ�
    if (n == 1)
        return;
    else{
        for (int i = 0; i < n / 2; i++){
            a[i] += a[n - i - 1];
        }
        n = n / 2;
        recursion(n);
    }
}

void double_cycle(int n){//����ѭ��
    for (long long int m = n; m > 1; m /= 2){//logn������
        for (long long int i = 0; i < m / 2; i++){
            a[i] = a[i * 2] + a[i * 2 + 1];// ����Ԫ����������洢��������ǰ��
        }
    }// a[0]Ϊ���ս��
}


int main(){
    long long int n;
    long long counter;//�ظ�����
    double seconds;//��ʱ��
    long long head, tail, freq, here;
    init(N);
    for (n = 2; n <= 1048576; n = n * 2)
    {
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);//��ʼ��ʱ
        counter = 0;
        while (true)
        {
            QueryPerformanceCounter((LARGE_INTEGER*)&here);//��¼��ǰʱ�䣬�����С�Ͷ�ѭ������
            if ((here - head) / freq * 1000000.0 > 10) {
                break;
            }
            //ordinary(n);
            //multi(n);
            //recursion(n);
            double_cycle(n);
            counter++;
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);//������ʱ
        seconds = (tail - head) / freq * 1000000.0;//��λת����us
        cout << n << ' ' << counter << ' ' << seconds << ' ' << seconds / counter << endl;
    }
    return 0;
}