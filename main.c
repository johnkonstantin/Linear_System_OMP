#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define SCHEDULE_ARG static

#define N 8000
#define TAU 0.00016
#define EPS 1e-9
#define MAX_NUM_ITERATIONS 10000

int main(void) {
	double* A = (double*)malloc(N * N * sizeof(double));
	double* b = (double*)malloc(N * sizeof(double));
	double* x = (double*)malloc(N * sizeof(double));
	double* x_0 = (double*)malloc(N * sizeof(double));

	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			if (i == j) {
				A[i * N + j] = 2.0;
			}
			else {
				A[i * N + j] = 1.0;
			}
		}
	}
	for (size_t i = 0; i < N; ++i) {
		b[i] = N + 1;
	}
	for (size_t i = 0; i < N; ++i) {
		x_0[i] = -1 * 1e10;
	}

	double begin, end;
	size_t ii = 0;
	double* x_t = (double*)malloc(N * sizeof(double));
	memcpy(x, x_0, N * sizeof(double));
	double* Ax = (double*)malloc(N * sizeof(double));
	double* sub1 = (double*)malloc(N * sizeof(double));
	double* tauSub1 = (double*)malloc(N * sizeof(double));
	double err = 1e15;
	double metric_b = 0;
	double metric_sub = 0;
	for (size_t i = 0; i < N; ++i) {
		metric_b += b[i] * b[i];
	}
	metric_b = sqrt(metric_b);
	begin = omp_get_wtime();

	int flag = 0;

#pragma omp parallel private(ii)
	for (ii = 0; ii < MAX_NUM_ITERATIONS; ++ii) {
		if (flag) {
			continue;
		}
#pragma omp single
		{
			memcpy(x_t, x, N * sizeof(double));
			metric_sub = 0.0;
		}
#pragma omp for schedule(SCHEDULE_ARG)
		for (size_t i = 0; i < N; ++i) {
			Ax[i] = 0.0;
		}
#pragma omp for schedule(SCHEDULE_ARG)
		for (size_t i = 0; i < N; ++i) {
			for (size_t j = 0; j < N; ++j) {
				Ax[i] += A[i * N + j] * x_t[j];
			}
		}
#pragma omp for schedule(SCHEDULE_ARG)
		for (size_t i = 0; i < N; ++i) {
			sub1[i] = Ax[i] - b[i];
		}
#pragma omp for schedule(SCHEDULE_ARG)
		for (size_t i = 0; i < N; ++i) {
			tauSub1[i] = sub1[i] * TAU;
		}
#pragma omp for schedule(SCHEDULE_ARG)
		for (size_t i = 0; i < N; ++i) {
			x[i] = x_t[i] - tauSub1[i];
		}
#pragma omp for schedule(SCHEDULE_ARG)
		for (size_t i = 0; i < N; ++i) {
			Ax[i] = 0.0;
		}
#pragma omp for schedule(SCHEDULE_ARG)
		for (size_t i = 0; i < N; ++i) {
			for (size_t j = 0; j < N; ++j) {
				Ax[i] += A[i * N + j] * x[j];
			}
		}
#pragma omp for schedule(SCHEDULE_ARG)
		for (size_t i = 0; i < N; ++i) {
			sub1[i] = Ax[i] - b[i];
		}
#pragma omp for reduction(+:metric_sub) schedule(SCHEDULE_ARG)
		for (size_t i = 0; i < N; ++i) {
			metric_sub += sub1[i] * sub1[i];
		}
#pragma omp single
		{
			metric_sub = sqrt(metric_sub);
			err = metric_sub / metric_b;
			printf("err = %f\n", err);
			if (err < EPS) {
				flag = 1;
			}
		}
	}
	end = omp_get_wtime();
	printf("Time = %f seconds\n", end - begin);
	if (N <= 20) {
		for (size_t i = 0; i < N; ++i) {
			printf("x[%zu] = %f\n", i, x[i]);
		}
	}
	else {
		printf("x[0] = %f\n", x[0]);
	}


	free(A);
	free(b);
	free(x);
	free(x_0);
	free(x_t);
	free(Ax);
	free(sub1);
	free(tauSub1);
	return 0;
}

