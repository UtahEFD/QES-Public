#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/Matrix3.h"
#include "util/VectorMath.h"

TEST_CASE("vector math class test")
{

  std::cout << "======================================\n"
            << "testing vector math on CPU            \n"
            << "--------------------------------------"
            << std::endl;

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
  REQUIRE(std::abs(d.length() - sqrt(3.0f)) < 1E-6);

  Vector3<float> n = { 0.0f, 0.0f, 1.0f };
  Vector3<float> x = { 2.0f, 2.0f, -2.0f };
  x = x.reflect(n);
  REQUIRE(x[0] == 2.0f);
  REQUIRE(x[1] == 2.0f);
  REQUIRE(x[2] == 2.0f);

  std::cout << "testing iostream : " << x << std::endl;
  std::cout << "======================================" << std::endl;
}

TEST_CASE("vector math class speed test")
{
  std::cout << "======================================\n"
            << "speed test vector math on CPU         \n"
            << "--------------------------------------"
            << std::endl;


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

  REQUIRE(c[0] == z._1);
  REQUIRE(c[1] == z._2);
  REQUIRE(c[2] == z._3);

  std::cout << "======================================" << std::endl;
}
