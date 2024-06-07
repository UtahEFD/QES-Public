#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/Matrix3.h"
#include "util/VectorMath.h"

TEST_CASE("vector math class test")
{

  printf("======================================\n");
  printf("testing vector math\n");
  printf("--------------------------------------\n");

  Vector3<float> a(1.0f, 2.0f, 3.0f);
  Vector3<float> b(1.0f, 2.0f, 3.0f);
  REQUIRE(a == b);

  Vector3<float> c = a + b;
  REQUIRE(c[0] == 2.0f);
  REQUIRE(c[1] == 4.0f);
  REQUIRE(c[2] == 6.0f);

  c += a;
  REQUIRE(c[0] == 3.0f);
  REQUIRE(c[1] == 6.0f);
  REQUIRE(c[2] == 9.0f);

  c /= 3;
  REQUIRE(c == a);

  c *= 3;
  REQUIRE(c == 3 * a);
  REQUIRE(c == a * 3);

  c = a - b;
  REQUIRE(c[0] == 0.0f);
  REQUIRE(c[1] == 0.0f);
  REQUIRE(c[2] == 0.0f);

  REQUIRE(a * b == 14);
  REQUIRE(a.dot(b) == 14);

  Vector3<float> d = { 1.0f, 1.0f, 1.0f };
  REQUIRE(d.length() == sqrt(3.0f));
}

TEST_CASE("vector math class speed test")
{
  size_t test_length = 2E9;
  float l = 0;

  Vector3<float> a(1.0f, 2.0f, 3.0f);
  Vector3<float> b(1.0f, 2.0f, 3.0f);
  Vector3<float> c;

  auto cpuStartTime = std::chrono::high_resolution_clock::now();
  for (auto it = 0; it < test_length; ++it) {
    c = a + b;
    c = 3.0f * a + b;
    c = a;

    l = c.length();
    c /= l;
  }
  auto cpuEndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpuElapsed = cpuEndTime - cpuStartTime;
  std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";

  vec3 x, y, z;
  x = { 1.0f, 2.0f, 3.0f };
  y = { 1.0f, 2.0f, 3.0f };

  cpuStartTime = std::chrono::high_resolution_clock::now();
  for (auto it = 0; it < test_length; ++it) {
    z._1 = x._1 + y._1;
    z._2 = x._2 + y._2;
    z._3 = x._3 + y._3;

    z._1 = 3.0f * x._1 + y._1;
    z._2 = 3.0f * x._2 + y._2;
    z._3 = 3.0f * x._3 + y._3;

    z = x;

    l = VectorMath::length(z);

    z._1 = z._1 / l;
    z._2 = z._2 / l;
    z._3 = z._3 / l;
  }
  cpuEndTime = std::chrono::high_resolution_clock::now();
  cpuElapsed = cpuEndTime - cpuStartTime;
  std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";
}

#if 0
void test_PlumeGeneralData::testCPU(int length)
{

  mat3 tmp = { 1, 2, 3, 2, 1, 2, 3, 2, 1 };
  std::vector<mat3> A;
  A.resize(length, tmp);

  std::vector<vec3> b;
  b.resize(length, { 1.0, 1.0, 1.0 });

  std::vector<vec3> x;
  x.resize(length, { 0.0, 0.0, 0.0 });

  std::vector<mat3sym> tau;
  // tau.resize(length, { 1, 2, 3, 1, 2, 1 });
  tau.resize(length, { 1, 0, 3, 0, 0, 1 });
  std::vector<vec3> invar;
  invar.resize(length, { 0.0, 0.0, 0.0 });

  auto cpuStartTime = std::chrono::high_resolution_clock::now();
  for (auto it = 0; it < length; ++it) {
    bool tt = vectorMath::invert3(A[it]);
    vectorMath::matmult(A[it], b[it], x[it]);
  }

  for (auto it = 0; it < length; ++it) {
    vectorMath::makeRealizable(10e-4, tau[it]);
    vectorMath::calcInvariants(tau[it], invar[it]);
  }

  auto cpuEndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpuElapsed = cpuEndTime - cpuStartTime;
  std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";

  std::cout << A[0]._11 << " " << A[0]._12 << " " << A[0]._13 << std::endl;
  std::cout << A[0]._21 << " " << A[0]._22 << " " << A[0]._23 << std::endl;
  std::cout << A[0]._31 << " " << A[0]._32 << " " << A[0]._33 << std::endl;
  std::cout << x[0]._1 << " " << x[0]._2 << " " << x[0]._3 << std::endl;

  std::cout << std::endl;

  std::cout << tau[0]._11 << " " << tau[0]._12 << " " << tau[0]._13 << std::endl;
  std::cout << tau[0]._12 << " " << tau[0]._22 << " " << tau[0]._23 << std::endl;
  std::cout << tau[0]._13 << " " << tau[0]._23 << " " << tau[0]._33 << std::endl;
  std::cout << invar[0]._1 << " " << invar[0]._2 << " " << invar[0]._3 << std::endl;

  return;
}
#endif