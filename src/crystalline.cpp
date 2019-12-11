#include <map>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "rectangular_lsap.h"
#include <cstdio>


static double _calculate_dsquared(double (*a)[3], double (*b)[3], double (*qoffset)[3]) {
	double total = 0;
	for (int i=0;i<3;i++) {
		double d = a[0][i] - b[0][i] - qoffset[0][i];
		total += d * d;
	}

	return total;
}

static int monoatomic_bipartite_matching(int num_atoms, int num_cells,
					double (*P)[3], double (*Q)[3], double (*nbr_cells)[3],
					double* p_cost, int64_t* permutation, double* distances)
{

	for (int i=0;i<num_atoms;i++) {
		for (int j=0;j<num_atoms;j++) {
			distances[i * num_atoms + j] = INFINITY;
		}
	}

	for (int k=0;k<num_cells;k++) {
		for (int i=0;i<num_atoms;i++) {
			for (int j=0;j<num_atoms;j++) {
				double dsq = _calculate_dsquared(&P[i], &Q[j], &nbr_cells[k]);
				distances[i * num_atoms + j] = std::min(dsq, distances[i * num_atoms + j]);
			}
		}
	}

	int res = solve_rectangular_linear_sum_assignment(num_atoms, num_atoms, distances, permutation);

	double cost = 0;
	for (int i=0;i<num_atoms;i++) {
		cost += distances[i * num_atoms + permutation[i]];
	}
	*p_cost += cost;
	return res;
}

static int _crystalline_bipartite_matching(int num_atoms, int num_cells,
						double (*P)[3], double (*Q)[3], double (*nbr_cells)[3],
						int* numbers, double* p_cost, int* permutation)
{
	std::vector< int64_t > _permutation(num_atoms, 0);
	std::vector< double > distances(num_atoms * num_atoms, INFINITY);

	std::map< int, int > start;
	std::map< int, int > count;
	int prev = -1;
	for (int i=0;i<num_atoms;i++) {
		int z = numbers[i];
		if (z != prev) {
			start[z] = i;
			count[z] = 0;
			prev = z;
		}
		count[z]++;
	}

	int res = 0;
	double cost = 0;
	std::map< int, int >::iterator it = start.begin();
	while (it != start.end()) {
		int z = it->first;
		int offset = start[z];
		int num = count[z];

		res = monoatomic_bipartite_matching(num, num_cells, &P[offset], &Q[offset], nbr_cells,
							&cost, &_permutation.data()[offset], distances.data());
		if (res != 0) {
			break;
		}

		it++;
	}

	for (int i=0;i<num_atoms;i++) {
		int z = numbers[i];
		int offset = start[z];
		permutation[i] = _permutation[i] + offset;
	}

	cost = sqrt(cost / num_atoms);
	*p_cost = cost;
	return res;
}

#ifdef __cplusplus
extern "C" {
#endif

int crystalline_bipartite_matching(int num_atoms, int num_cells,
        		           double* P, double* Q, double* nbr_cells, int* numbers,
         		          double* cost, int* permutation)
{
	return _crystalline_bipartite_matching(num_atoms, num_cells,
						(double (*)[3])P, (double (*)[3])Q, (double (*)[3])nbr_cells,
						numbers, cost, permutation);
}

#ifdef __cplusplus
}
#endif
