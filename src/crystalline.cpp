#include <cmath>
#include <vector>
#include "rectangular_lsap.h"


static double calculate_dsquared(double (*a)[3], double (*b)[3]) {
	double total = 0;
	for (int i=0;i<3;i++) {
		double d = a[0][i] - b[0][i];
		total += d * d;
	}

	return total;
}

static int _crystalline_bipartite_matching(int num_atoms, int num_cells,
                                          double (*P)[3], double (*Q)[3],
                                          double* p_cost, int* permutation)
{
	std::vector< int64_t > _permutation(num_atoms, 0);
	std::vector< double > distances(num_atoms * num_atoms, INFINITY);

	for (int k=0;k<num_cells;k++) {
		for (int i=0;i<num_atoms;i++) {
			for (int j=0;j<num_atoms;j++) {
				double dsq = calculate_dsquared(&P[i], &Q[k * num_atoms + j]);
				distances[i * num_atoms + j] = std::min(dsq, distances[i * num_atoms + j]);
			}
		}
	}

	int res = solve_rectangular_linear_sum_assignment(num_atoms, num_atoms, distances.data(),
								_permutation.data());

	double cost = 0;
	for (int i=0;i<num_atoms;i++) {
		permutation[i] = _permutation[i];
		cost += distances[i * num_atoms + permutation[i]];
	}

	*p_cost = cost;
	return res;
}

#ifdef __cplusplus
extern "C" {
#endif

int crystalline_bipartite_matching(int num_atoms, int num_cells,
                                   double* P, double* Q,
                                   double* cost, int* permutation)
{
	return _crystalline_bipartite_matching(num_atoms, num_cells, (double (*)[3])P, (double (*)[3])Q,
						cost, permutation);
}

#ifdef __cplusplus
}
#endif
