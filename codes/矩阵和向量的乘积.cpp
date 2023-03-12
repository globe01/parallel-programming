//矩阵与向量乘积

#include <iostream>
#include <windows.h>
using namespace std;
const int N = 10000;//问题规模
double b[N][N], a[N], sum[N];

void init(int N) {
    for (int i = 0; i < N; i++) {//初始化给定向量a[i]=i
        a[i] = i;
    }
    for (int i = 0; i < N; i++) {//初始化给定矩阵b[i][j]=i+j
        for (int j = 0; j < N; j++) {
            b[i][j] = i + j;
        }
    }
}

void ordinary(int N) {//平凡算法 逐列访问，一步外层循环（内存循环一次完整执行）计算出一个内积结果
    for (int i = 0; i < N; i++) {
        sum[i] = 0.0;
        for (int j = 0; j < N; j++){
            sum[i] += b[j][i] * a[j];
        }
    }
}

void cache(int N) {//优化算法 改为逐行访问矩阵元素，一步外层循环计算不出任何一个内积，只是向每个内积累加一个乘法结果
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
    long counter;//重复次数
    double seconds;//总时间
    long long head, tail, freq, here;
    init(N);
    for (n = 0; n <= 10000; n += step){
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
        counter = 0;
        while (true)
        {
            QueryPerformanceCounter((LARGE_INTEGER*)&here);//记录当前时间，如果很小就多重复几次
            if ((here - head) / freq * 1000.0 > 10) {
                break;
            }
            ordinary(n);
            //cache(n);
            counter++;
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
        seconds = (tail - head) / freq *1000.0;//单位转化成ms
        cout << n << ' ' << counter << ' ' << seconds << ' ' << seconds / counter << endl;
        if (n == 1000)
            step = 1000;
    }
    return 0;
}