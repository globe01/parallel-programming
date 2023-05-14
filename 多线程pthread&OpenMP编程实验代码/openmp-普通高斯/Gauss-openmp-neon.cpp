//
////ע�����ļ�����Źؼ����ִ���
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
//		//���в���
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
//		//���в���
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
//		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
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
//		//���в���
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
//		//���в���
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
//		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
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
