#include <map>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include "rectangular_lsap.h"


static double calculate_dsquared(double (*a)[3], double (*b)[3], int num_cells, double (*nbr_cells)[3]) {

	double best = INFINITY;
	for (int j=0;j<num_cells;j++) {

		double dx = a[0][0] - b[0][0] - nbr_cells[j][0];
		double dy = a[0][1] - b[0][1] - nbr_cells[j][1];
		double dz = a[0][2] - b[0][2] - nbr_cells[j][2];

		double dsq = dx * dx + dy * dy + dz * dz;
		best = std::min(best, dsq);
	}

	return best;
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


	for (int i=0;i<num_atoms;i++) {
		for (int j=0;j<num_atoms;j++) {
			double dsq = calculate_dsquared(&P[i], &Q[j], num_cells, nbr_cells);
			distances[i * num_atoms + j] = std::min(dsq, distances[i * num_atoms + j]);
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

int crystalline_bipartite_matching(int num_atoms, int num_cells,
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

