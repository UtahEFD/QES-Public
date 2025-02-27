#include <chrono>
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"

void test_matrix_multiplication_gpu(const int &length, std::vector<mat3> &A, std::vector<vec3> &b, std::vector<vec3> &x);
void test_matrix_inversion_gpu(const int &length, std::vector<mat3> &A);
void test_matrix_invariants_gpu(const int &length, std::vector<mat3sym> &A, std::vector<vec3> &x);
