#include <map>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <cstring>
#include "lup_decomposition.h"
#include "matrix_vector.h"


static double wrap(double x) {
	x = fmod(x, 1);
	if (x < 0) {
		x += 1;
	}
	if (x == -0) x = 0;

	if (x == 1) {
		return 0;
	}
	else {
		return x;
	}
}

int _wrap_positions(int num_atoms, double (*P)[3], double (*cell)[3], int8_t* pbc, double (*wrapped)[3]) {

	double transposed[9];
	memcpy(transposed, cell, 9 * sizeof(double));
	transpose(3, transposed);

	int pivot[3];
	double LU[9] = {0};
	memcpy(LU, transposed, 9 * sizeof(double));
	lup_decompose(3, LU, pivot);	//todo: check return code

	for (int i=0;i<num_atoms;i++) {
		double scaled[3];
		lup_solve(3, LU, pivot, P[i], scaled);

		for (int j=0;j<3;j++) {
			if (pbc[j]) {
				scaled[j] = wrap(scaled[j]);
			}
		}

		matvec(3, transposed, scaled, wrapped[i]);
	}

	return 0;
}

