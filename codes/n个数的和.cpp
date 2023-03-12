//n个数求和
#include <iostream>
#include <windows.h>
using namespace std;
const long long int N = 1048576;//2的20次方
long long int sum;
double a[N];

void init(int n){//初始化
    for (long long int i = 0; i < N; i++){
        a[i] = i;
    }
}

void ordinary(int n){//平凡算法
    for (int i = 0; i < n; i++){
        sum += a[i];
    }
}

void multi(int n){//多路链式
    long long int sum1 = 0, sum2 = 0;
    for (long long int i = 0; i < n; i += 2){//步长为2
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;
}

void recursion(int n){//递归函数，优点是简单，缺点是递归函数调用开销较大
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

void double_cycle(int n){//二重循环
    for (long long int m = n; m > 1; m /= 2){//logn个步骤
        for (long long int i = 0; i < m / 2; i++){
            a[i] = a[i * 2] + a[i * 2 + 1];// 相邻元素相加连续存储到数组最前面
        }
    }// a[0]为最终结果
}


int main(){
    long long int n;
    long long counter;//重复次数
    double seconds;//总时间
    long long head, tail, freq, here;
    init(N);
    for (n = 2; n <= 1048576; n = n * 2)
    {
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
        counter = 0;
        while (true)
        {
            QueryPerformanceCounter((LARGE_INTEGER*)&here);//记录当前时间，如果很小就多循环几次
            if ((here - head) / freq * 1000000.0 > 10) {
                break;
            }
            //ordinary(n);
            //multi(n);
            //recursion(n);
            double_cycle(n);
            counter++;
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
        seconds = (tail - head) / freq * 1000000.0;//单位转化成us
        cout << n << ' ' << counter << ' ' << seconds << ' ' << seconds / counter << endl;
    }
    return 0;
}