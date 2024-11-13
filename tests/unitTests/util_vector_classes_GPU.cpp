#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/Matrix3.h"
#include "util/VectorMath.h"
#include "CUDA_vector_testkernel.h"

const int length = 1E6;

TEST_CASE("Matrix Multiplication")
{
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

  for (size_t k = 0; k < x_array.size(); ++k) {
    REQUIRE_THAT(x_array[k]._1, Catch::Matchers::WithinRel(6.0f, 0.000001f));
    REQUIRE_THAT(x_array[k]._2, Catch::Matchers::WithinRel(15.0f, 0.000001f));
    REQUIRE_THAT(x_array[k]._3, Catch::Matchers::WithinRel(24.0F, 0.000001f));
  }
}

TEST_CASE("Matrix Inversion")
{
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Matrix Inversion" << std::endl;

  mat3 A = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  std::vector<mat3> A_array;
  A_array.resize(length, A);

  test_matrix_inversion_gpu(length, A_array);

  float eps = 1.0e-3;
  for (size_t k = 0; k < A_array.size(); ++k) {
      REQUIRE_THAT(A_array[k]._11, Catch::Matchers::WithinAbs(1.0F, eps));
      REQUIRE_THAT(A_array[k]._12, Catch::Matchers::WithinRel(0.0f, eps));
      REQUIRE_THAT(A_array[k]._13, Catch::Matchers::WithinRel(0.0f, eps));

      REQUIRE_THAT(A_array[k]._21, Catch::Matchers::WithinAbs(0.0F, eps));
      REQUIRE_THAT(A_array[k]._22, Catch::Matchers::WithinRel(1.0f, eps));
      REQUIRE_THAT(A_array[k]._23, Catch::Matchers::WithinRel(0.0f, eps));

      REQUIRE_THAT(A_array[k]._31, Catch::Matchers::WithinAbs(0.0F, eps));
      REQUIRE_THAT(A_array[k]._32, Catch::Matchers::WithinRel(0.0f, eps));
      REQUIRE_THAT(A_array[k]._33, Catch::Matchers::WithinRel(1.0f, eps));
  }


  mat3 D = { 1.0f, 2.0f, -1.0f, 2.0f, 1.0f, 2.0f, -1.0f, 2.0f, 1.0f };
  std::vector<mat3> D_array;
  D_array.resize(length, D);

  test_matrix_inversion_gpu(length, D_array);

  SECTION("D Array Test");
  
  eps = 1.0e-4;
  for (size_t k = 0; k < D_array.size(); ++k) {
    REQUIRE_THAT(D_array[k]._11, Catch::Matchers::WithinRel(3.0f/16.0f, eps));
    REQUIRE_THAT(D_array[k]._12, Catch::Matchers::WithinRel(1.0f/4.0f, eps));
    REQUIRE_THAT(D_array[k]._13, Catch::Matchers::WithinRel(-5.0f/16.0f, eps));

    REQUIRE_THAT(D_array[k]._21, Catch::Matchers::WithinRel(1.0f/4.0f, eps));
    REQUIRE_THAT(D_array[k]._22, Catch::Matchers::WithinRel(0.0f, eps));
    REQUIRE_THAT(D_array[k]._23, Catch::Matchers::WithinRel(1.0f/4.0f, eps));

    REQUIRE_THAT(D_array[k]._31, Catch::Matchers::WithinRel(-5.0f/16.0f, eps));
    REQUIRE_THAT(D_array[k]._32, Catch::Matchers::WithinRel(1.0f/4.0f, eps));
    REQUIRE_THAT(D_array[k]._33, Catch::Matchers::WithinRel(3.0f/16.0f, eps));
  }


  mat3 C = { 0.5, -1.2, 0.8, -0.3, 2.1, -0.4, 1.3, -0.7, 1.9 };
  std::vector<mat3> C_array;
  C_array.resize(length, C);

  test_matrix_inversion_gpu(length, C_array);

  SECTION("C Array Test");
  
  eps = 1.0e-4;
  for (size_t k = 0; k < C_array.size(); ++k) {
    REQUIRE_THAT(C_array[k]._11, Catch::Matchers::WithinRel(-16.7873f, eps));
    REQUIRE_THAT(C_array[k]._12, Catch::Matchers::WithinRel(-7.7828f, eps));
    REQUIRE_THAT(C_array[k]._13, Catch::Matchers::WithinRel(5.4299f, eps));

    REQUIRE_THAT(C_array[k]._21, Catch::Matchers::WithinRel(-0.22624f, eps));
    REQUIRE_THAT(C_array[k]._22, Catch::Matchers::WithinRel(0.4072f, eps));
    REQUIRE_THAT(C_array[k]._23, Catch::Matchers::WithinRel(0.1810f, eps));

    REQUIRE_THAT(C_array[k]._31, Catch::Matchers::WithinRel(11.4027f, eps));
    REQUIRE_THAT(C_array[k]._32, Catch::Matchers::WithinRel(5.4752f, eps));
    REQUIRE_THAT(C_array[k]._33, Catch::Matchers::WithinRel(-3.1222f, eps));
  }

  
  SECTION("B Array Test");

  mat3 B = { 1, 2, 3, 2, 1, 2, 3, 2, 1 };
  std::vector<mat3> B_array;
  B_array.resize(length, B);

  test_matrix_inversion_gpu(length, B_array);

  for (size_t k = 0; k < B_array.size(); ++k) {
    REQUIRE_THAT(B_array[k]._11, Catch::Matchers::WithinRel(-0.375f, 0.000001f));
    REQUIRE_THAT(B_array[k]._12, Catch::Matchers::WithinRel(0.500f, 0.000001f));
    REQUIRE_THAT(B_array[k]._13, Catch::Matchers::WithinRel(0.125f, 0.000001f));

    REQUIRE_THAT(B_array[k]._21, Catch::Matchers::WithinRel(0.500f, 0.000001f));
    REQUIRE_THAT(B_array[k]._22, Catch::Matchers::WithinRel(-1.000f, 0.000001f));
    REQUIRE_THAT(B_array[k]._23, Catch::Matchers::WithinRel(0.500f, 0.000001f));

    REQUIRE_THAT(B_array[k]._31, Catch::Matchers::WithinRel(0.125f, 0.000001f));
    REQUIRE_THAT(B_array[k]._32, Catch::Matchers::WithinRel(0.500f, 0.000001f));
    REQUIRE_THAT(B_array[k]._33, Catch::Matchers::WithinRel(-0.375f, 0.000001f));
  }
}

TEST_CASE("Symmetric Matrix Invariants")
{
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Symmetric Matrix Invariants" << std::endl;

  mat3sym tau = { 1, 2, 3, 1, 2, 1 };
  std::vector<mat3sym> tau_array;
  tau_array.resize(length, tau);
  std::vector<vec3> x_array;
  x_array.resize(length, { 0, 0, 0 });

  test_matrix_invariants_gpu(length, tau_array, x_array);

  for (size_t k = 0; k < x_array.size(); ++k) {
    REQUIRE_THAT(x_array[k]._1, Catch::Matchers::WithinRel(3.0f, 0.000001f));
    REQUIRE_THAT(x_array[k]._2, Catch::Matchers::WithinRel(-14.0f, 0.000001f));
    REQUIRE_THAT(x_array[k]._3, Catch::Matchers::WithinRel(8.0f, 0.000001f));
  }
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
