#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
  REQUIRE_THAT(c[0], Catch::Matchers::WithinRel(2.0F, 0.01F));
  REQUIRE_THAT(c[1], Catch::Matchers::WithinRel(4.0F, 0.01F));
  REQUIRE_THAT(c[2], Catch::Matchers::WithinRel(6.0F, 0.01F));

  c += a;
  REQUIRE_THAT(c[0], Catch::Matchers::WithinRel(3.0F, 0.01F));
  REQUIRE_THAT(c[1], Catch::Matchers::WithinRel(6.0F, 0.01F));
  REQUIRE_THAT(c[2], Catch::Matchers::WithinRel(9.0F, 0.01F));

  c /= 3;
  REQUIRE(c == a);

  c *= 3;
  REQUIRE(c == 3 * a);
  REQUIRE(c == a * 3);

  c = a - b;
  REQUIRE_THAT(c[0], Catch::Matchers::WithinRel(0.0F, 0.01F));
  REQUIRE_THAT(c[1], Catch::Matchers::WithinRel(0.0F, 0.01F));
  REQUIRE_THAT(c[2], Catch::Matchers::WithinRel(0.0F, 0.01F));

  REQUIRE(a * b == 14);
  REQUIRE(a.dot(b) == 14);

  Vector3<float> d = { 1.0f, 1.0f, 1.0f };
  REQUIRE(std::abs(d.length() - sqrt(3.0f)) < 1E-6);

  Vector3<float> n = { 0.0f, 0.0f, 1.0f };
  Vector3<float> x = { 2.0f, 2.0f, -2.0f };
  x = x.reflect(n);

  REQUIRE_THAT(x[0], Catch::Matchers::WithinRel(2.0F, 0.01F));
  REQUIRE_THAT(x[1], Catch::Matchers::WithinRel(2.0F, 0.01F));
  REQUIRE_THAT(x[2], Catch::Matchers::WithinRel(2.0F, 0.01F));

  std::cout << "testing iostream : " << x << std::endl;
  std::cout << "======================================" << std::endl;
}

TEST_CASE("vector math class speed test")
{
  std::cout << "======================================\n"
            << "speed test vector math on CPU         \n"
            << "--------------------------------------"
            << std::endl;

  auto test_length = 2E7;
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

  REQUIRE_THAT(c[0], Catch::Matchers::WithinRel(z._1, 0.01F));
  REQUIRE_THAT(c[1], Catch::Matchers::WithinRel(z._2, 0.01F));
  REQUIRE_THAT(c[2], Catch::Matchers::WithinRel(z._3, 0.01F));

  std::cout << "======================================" << std::endl;
}
