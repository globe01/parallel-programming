#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <CL/sycl.hpp>
#include <chrono>

using namespace std;
using namespace cl::sycl;

constexpr size_t N = 1024;  // 矩阵大小或数组长度

// 定义全局内存缓冲区
buffer<int, 2> Core_buf((range<2>(N, N)));
buffer<int, 2> Rows_buf(range<2>(N, N));

// 读取消元子并初始化
void Core_reset(queue& q) {
    ifstream infile("Core_8.txt");
    char fin[10000] = { 0 };
    int a, curr;
    while (infile.getline(fin, sizeof(fin))) {
        std::stringstream line(fin);
        int p = 0;
        while (line >> a) {
            if (p == 0) {
                curr = a;
                p = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            auto Core_acc = Core_buf.get_access<access::mode::write>(q);
            Core_acc[curr][N - 1 - j] += temp;
            Core_acc[curr][N] = 1;  // 1非空 0空
        }
    }
}

// 读取被消元行并初始化
void Rows_reset(queue& q) {
    ifstream infile("Rows_8.txt");
    char fin[10000] = { 0 };
    int a, curr = 0;
    while (infile.getline(fin, sizeof(fin))) {
        std::stringstream line(fin);
        int p = 0;
        while (line >> a) {
            if (p == 0) {
                auto Rows_acc = Rows_buf.get_access<access::mode::write>(q);
                Rows_acc[curr][N] = a;
                p = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            auto Rows_acc = Rows_buf.get_access<access::mode::write>(q);
            Rows_acc[curr][N - 1 - j] += temp;
        }
        curr++;
    }
}

// 串行算法
void Serial() {
    // 获取 Core 和 Rows 的访问器
    auto Core_acc = Core_buf.get_access<access::mode::read_write>();
    auto Rows_acc = Rows_buf.get_access<access::mode::read_write>();

    int column = N * 32;
    int each_col = 32;
    int Rows_num = N;

    for (int i = column - 1; i - each_col >= -1; i -= each_col) {  // 遍历消元子的列
        for (int j = 0; j < Rows_num; j++) {  // 遍历被消元行
            while (Rows_acc[j][N] <= i && Rows_acc[j][N] >= i - each_col + 1) {  // 若非零元素在当前消元子处理范围内
                int curr = Rows_acc[j][N];
                if (Core_acc[curr][N] == 1) {  // 消元子非空时进行消去，即异或，异或之后找新的首个非零元素
                    for (int k = 0; k < N; k++) {
                        Rows_acc[j][k] = Rows_acc[j][k] ^ Core_acc[curr][k];  // 这一循环部分后续可进行SIMD优化
                    }
                    int t = 0;
                    for (int m = 0; m < N; m++) {
                        if (Rows_acc[j][m] != 0) {
                            unsigned int temp = Rows_acc[j][m];
                            while (temp != 0) {
                                temp = temp >> 1;
                                t++;
                            }
                            t += (column - m - 1) * 32;
                            break;
                        }
                    }
                    Rows_acc[j][N] = t - 1;
                }
                else {  // 消元子为空，被消元行升格为消元子
                    for (int k = 0; k < N; k++) {
                        Core_acc[curr][k] = Rows_acc[j][k];
                    }
                    Core_acc[curr][N] = 1;
                    break;
                }
            }
        }
    }
}

// 并行算法
void Parallel(queue& q) {
    // 获取 Core 和 Rows 的访问器
    auto Core_acc = Core_buf.get_access<access::mode::read_write>(q);
    auto Rows_acc = Rows_buf.get_access<access::mode::read_write>(q);

    int column = N * 32;
    int each_col = 32;
    int Rows_num = N;

    for (int i = column - 1; i - each_col >= -1; i -= each_col) {  // 遍历消元子的列
        for (int j = 0; j < Rows_num; j++) {  // 遍历被消元行
            while (Rows_acc[j][N] <= i && Rows_acc[j][N] >= i - each_col + 1) {  // 若非零元素在当前消元子处理范围内
                int curr = Rows_acc[j][N];
                if (Core_acc[curr][N] == 1) {  // 消元子非空时进行消去，即异或，异或之后找新的首个非零元素
                    for (int k = 0; k < N; k++) {
                        Rows_acc[j][k] = Rows_acc[j][k] ^ Core_acc[curr][k];  // 这一循环部分后续可进行SIMD优化
                    }
                    int t = 0;
                    for (int m = 0; m < N; m++) {
                        if (Rows_acc[j][m] != 0) {
                            unsigned int temp = Rows_acc[j][m];
                            while (temp != 0) {
                                temp = temp >> 1;
                                t++;
                            }
                            t += (column - m - 1) * 32;
                            break;
                        }
                    }
                    Rows_acc[j][N] = t - 1;
                }
                else {  // 消元子为空，被消元行升格为消元子
                    for (int k = 0; k < N; k++) {
                        Core_acc[curr][k] = Rows_acc[j][k];
                    }
                    Core_acc[curr][N] = 1;
                    break;
                }
            }
        }
    }
}

int main() {
    queue q(es::gpu_selector{});

    // 重置并初始化 Core 和 Rows
    Core_reset(q);
    Rows_reset(q);

    // 串行算法计时
    auto start_serial = chrono::high_resolution_clock::now();
    Serial();
    auto end_serial = chrono::high_resolution_clock::now();
    chrono::duration<double> serial_duration = end_serial - start_serial;

    cout << "Serial Time: " << serial_duration.count() << " seconds" << endl;

    // 并行算法计时
    auto start_parallel = chrono::high_resolution_clock::now();
    Parallel(q);
    q.wait();
    auto end_parallel = chrono::high_resolution_clock::now();
    chrono::duration<double> parallel_duration = end_parallel - start_parallel;

    cout << "Parallel Time: " << parallel_duration.count() << " seconds" << endl;

    return 0;
}
