#ifndef WRAP_POSITIONS_H
#define WRAP_POSITIONS_H

int _wrap_positions(int num_atoms, double (*P)[3], double (*cell)[3], int8_t* pbc, double (*wrapped)[3]);

#endif
