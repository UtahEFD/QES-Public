#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/Matrix3.h"
#include "util/VectorMath.h"
#include "CUDA_vector_testkernel.h"

TEST_CASE("vector math class test")
{

  std::cout << "======================================\n"
            << "testing vector math\n"
            << "--------------------------------------" << std::endl;

  int length = 1E6;

  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Matrix Multiplication" << std::endl;
  mat3 A = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  vec3 b = { 1, 1, 1 };

  std::vector<mat3> A_array;
  A_array.resize(length, A);

  std::vector<vec3> b_array;
  b_array.resize(length, b);

  std::vector<vec3> x_array;
  x_array.resize(length, { 0, 0, 0 });

  test_matrix_multiplication_gpu(length, A_array, b_array, x_array);

  REQUIRE(x_array[0]._1 == 6);
  REQUIRE(x_array[0]._2 == 15);
  REQUIRE(x_array[0]._3 == 24);

  vec3 sum = { 0.0, 0.0, 0.0 };
  for (size_t k = 0; k < x_array.size(); ++k) {
    sum._1 += x_array[k]._1;
    sum._2 += x_array[k]._2;
    sum._3 += x_array[k]._3;
  }
  REQUIRE(sum._1 == 6 * length);
  REQUIRE(sum._2 == 15 * length);
  REQUIRE(sum._3 == 24 * length);

  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Matrix Inversion" << std::endl;

  mat3 B = { 1, 2, 3, 2, 1, 2, 3, 2, 1 };
  std::vector<mat3> B_array;
  B_array.resize(length, B);

  test_matrix_inversion_gpu(length, B_array);

  REQUIRE(B_array[0]._11 == -0.375);
  REQUIRE(B_array[0]._12 == 00.500);
  REQUIRE(B_array[0]._13 == 00.125);

  REQUIRE(B_array[0]._21 == 00.500);
  REQUIRE(B_array[0]._22 == -1.000);
  REQUIRE(B_array[0]._23 == 00.500);

  REQUIRE(B_array[0]._31 == 00.125);
  REQUIRE(B_array[0]._32 == 00.500);
  REQUIRE(B_array[0]._33 == -0.375);


  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Symmetric Matrix Invariants" << std::endl;

  mat3sym tau = { 1, 2, 3, 1, 2, 1 };
  std::vector<mat3sym> tau_array;
  tau_array.resize(length, tau);
  x_array.resize(length, { 0, 0, 0 });

  test_matrix_invariants_gpu(length, tau_array, x_array);

  REQUIRE(x_array[0]._1 == 3.0);
  REQUIRE(x_array[0]._2 == -14.0);
  REQUIRE(x_array[0]._3 == 8.0);

  sum = { 0.0, 0.0, 0.0 };
  for (size_t k = 0; k < x_array.size(); ++k) {
    sum._1 += x_array[k]._1;
    sum._2 += x_array[k]._2;
    sum._3 += x_array[k]._3;
  }
  REQUIRE(sum._1 == 3.0 * length);
  REQUIRE(sum._2 == -14.0 * length);
  REQUIRE(sum._3 == 8.0 * length);
  std::cout << "======================================" << std::endl;
}

/*
std::cout << "--------------------------------------" << std::endl;
std::cout << "Sample of calculations" << std::endl;

std::cout << A[0]._11 << " " << A[0]._12 << " " << A[0]._13 << std::endl;
std::cout << A[0]._21 << " " << A[0]._22 << " " << A[0]._23 << std::endl;
std::cout << A[0]._31 << " " << A[0]._32 << " " << A[0]._33 << std::endl;

std::cout << std::endl;

std::cout << x[0]._1 << " " << x[0]._2 << " " << x[0]._3 << std::endl;

std::cout << std::endl;

std::cout << tau[0]._11 << " " << tau[0]._12 << " " << tau[0]._13 << std::endl;
std::cout << tau[0]._12 << " " << tau[0]._22 << " " << tau[0]._23 << std::endl;
std::cout << tau[0]._13 << " " << tau[0]._23 << " " << tau[0]._33 << std::endl;

std::cout << std::endl;

std::cout << invar[0]._1 << " " << invar[0]._2 << " " << invar[0]._3 << std::endl;
std::cout << "--------------------------------------" << std::endl;
*/
