//
////注：本文件仅存放关键部分代码
//
//
//void Static_neon()
//{
//	float32x4_t va = vmovq_n_f32(0);
//	float32x4_t vx = vmovq_n_f32(0);
//	float32x4_t vaij = vmovq_n_f32(0);
//	float32x4_t vaik = vmovq_n_f32(0);
//	float32x4_t vakj = vmovq_n_f32(0);
//
//#pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
//	for (int k = 0; k < N; k++)
//	{
//		//串行部分
//#pragma omp single
//		{
//			float32x4_t vt = vmovq_n_f32(M[k][k]);
//			int j;
//			for (j = k + 1; j < N; j++)
//			{
//				va = vld1q_f32(&(M[k][j]));
//				va = vdivq_f32(va, vt);
//				vst1q_f32(&(M[k][j]), va);
//			}
//			for (; j < N; j++)
//			{
//				M[k][j] = M[k][j] * 1.0 / M[k][k];
//
//			}
//			M[k][k] = 1.0;
//		}
//
//		//并行部分
//#pragma omp for schedule(Static)
//		for (int i = k + 1; i < N; i++)
//		{
//			vaik = vmovq_n_f32(M[i][k]);
//			int j;
//			for (j = k + 1; j + 4 <= N; j += 4)
//			{
//				vakj = vld1q_f32(&(M[k][j]));
//				vaij = vld1q_f32(&(M[i][j]));
//				vx = vmulq_f32(vakj, vaik);
//				vaij = vsubq_f32(vaij, vx);
//
//				vst1q_f32(&M[i][j], vaij);
//			}
//
//			for (; j < N; j++)
//			{
//				M[i][j] = M[i][j] - M[i][k] * M[k][j];
//			}
//
//			M[i][k] = 0;
//		}
//		// 离开for循环时，各个线程默认同步，进入下一行的处理
//	}
//}
//
//
//void Dynamic_neon()
//{
//	float32x4_t va = vmovq_n_f32(0);
//	float32x4_t vx = vmovq_n_f32(0);
//	float32x4_t vaij = vmovq_n_f32(0);
//	float32x4_t vaik = vmovq_n_f32(0);
//	float32x4_t vakj = vmovq_n_f32(0);
//
//	#pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
//	for (int k = 0; k < N; k++)
//	{
//		//串行部分
//		#pragma omp single
//		{
//			float32x4_t vt=vmovq_n_f32(M[k][k]);
//			int j;
//			for (j = k + 1; j < N; j++)
//			{
//				va=vld1q_f32(&(M[k][j]) );
//				va= vdivq_f32(va,vt);
//				vst1q_f32(&(M[k][j]), va);
//			}
//			for(; j<N; j++)
//			{
//				M[k][j]=M[k][j]*1.0 / M[k][k];
//
//			}
//			M[k][k] = 1.0;
//		}
//
//		//并行部分
//		#pragma omp for schedule(dynamic, 5)
//		for (int i = k + 1; i < N; i++)
//		{
//			vaik=vmovq_n_f32(M[i][k]);
//			int j;
//			for (j = k + 1; j+4 <= N; j+=4)
//			{
//				vakj=vld1q_f32(&(M[k][j]));
//				vaij=vld1q_f32(&(M[i][j]));
//				vx=vmulq_f32(vakj,vaik);
//				vaij=vsubq_f32(vaij,vx);
//
//				vst1q_f32(&M[i][j], vaij);
//			}
//
//			for(; j<N; j++)
//			{
//				M[i][j] = M[i][j] - M[i][k] * M[k][j];
//			}
//
//			M[i][k] = 0;
//		}
//		// 离开for循环时，各个线程默认同步，进入下一行的处理
//	}
//}
//
//
//
//
//ReStart();
//QueryPerformanceCounter((LARGE_INTEGER *)&head);
//Static_neon();
//QueryPerformanceCounter((LARGE_INTEGER *)&tail );
//seconds = (tail - head) * 1000.0 / freq ;
//cout << "Static_neon: " << seconds << "ms" << endl;
//
//
//
//ReStart();
//QueryPerformanceCounter((LARGE_INTEGER *)&head);
//Dynamic_neon();
//QueryPerformanceCounter((LARGE_INTEGER *)&tail );
//seconds = (tail - head) * 1000.0 / freq ;
//cout << "Dynamic_neon: " << seconds << "ms" << endl;
//
//
