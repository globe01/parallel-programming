#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <CL/sycl.hpp>
#include <chrono>

using namespace std;
using namespace cl::sycl;

constexpr size_t N = 1024;  // �����С�����鳤��

// ����ȫ���ڴ滺����
buffer<int, 2> Core_buf((range<2>(N, N)));
buffer<int, 2> Rows_buf(range<2>(N, N));

// ��ȡ��Ԫ�Ӳ���ʼ��
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
            Core_acc[curr][N] = 1;  // 1�ǿ� 0��
        }
    }
}

// ��ȡ����Ԫ�в���ʼ��
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

// �����㷨
void Serial() {
    // ��ȡ Core �� Rows �ķ�����
    auto Core_acc = Core_buf.get_access<access::mode::read_write>();
    auto Rows_acc = Rows_buf.get_access<access::mode::read_write>();

    int column = N * 32;
    int each_col = 32;
    int Rows_num = N;

    for (int i = column - 1; i - each_col >= -1; i -= each_col) {  // ������Ԫ�ӵ���
        for (int j = 0; j < Rows_num; j++) {  // ��������Ԫ��
            while (Rows_acc[j][N] <= i && Rows_acc[j][N] >= i - each_col + 1) {  // ������Ԫ���ڵ�ǰ��Ԫ�Ӵ���Χ��
                int curr = Rows_acc[j][N];
                if (Core_acc[curr][N] == 1) {  // ��Ԫ�ӷǿ�ʱ������ȥ����������֮�����µ��׸�����Ԫ��
                    for (int k = 0; k < N; k++) {
                        Rows_acc[j][k] = Rows_acc[j][k] ^ Core_acc[curr][k];  // ��һѭ�����ֺ����ɽ���SIMD�Ż�
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
                else {  // ��Ԫ��Ϊ�գ�����Ԫ������Ϊ��Ԫ��
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

// �����㷨
void Parallel(queue& q) {
    // ��ȡ Core �� Rows �ķ�����
    auto Core_acc = Core_buf.get_access<access::mode::read_write>(q);
    auto Rows_acc = Rows_buf.get_access<access::mode::read_write>(q);

    int column = N * 32;
    int each_col = 32;
    int Rows_num = N;

    for (int i = column - 1; i - each_col >= -1; i -= each_col) {  // ������Ԫ�ӵ���
        for (int j = 0; j < Rows_num; j++) {  // ��������Ԫ��
            while (Rows_acc[j][N] <= i && Rows_acc[j][N] >= i - each_col + 1) {  // ������Ԫ���ڵ�ǰ��Ԫ�Ӵ���Χ��
                int curr = Rows_acc[j][N];
                if (Core_acc[curr][N] == 1) {  // ��Ԫ�ӷǿ�ʱ������ȥ����������֮�����µ��׸�����Ԫ��
                    for (int k = 0; k < N; k++) {
                        Rows_acc[j][k] = Rows_acc[j][k] ^ Core_acc[curr][k];  // ��һѭ�����ֺ����ɽ���SIMD�Ż�
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
                else {  // ��Ԫ��Ϊ�գ�����Ԫ������Ϊ��Ԫ��
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

    // ���ò���ʼ�� Core �� Rows
    Core_reset(q);
    Rows_reset(q);

    // �����㷨��ʱ
    auto start_serial = chrono::high_resolution_clock::now();
    Serial();
    auto end_serial = chrono::high_resolution_clock::now();
    chrono::duration<double> serial_duration = end_serial - start_serial;

    cout << "Serial Time: " << serial_duration.count() << " seconds" << endl;

    // �����㷨��ʱ
    auto start_parallel = chrono::high_resolution_clock::now();
    Parallel(q);
    q.wait();
    auto end_parallel = chrono::high_resolution_clock::now();
    chrono::duration<double> parallel_duration = end_parallel - start_parallel;

    cout << "Parallel Time: " << parallel_duration.count() << " seconds" << endl;

    return 0;
}
