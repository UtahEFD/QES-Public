#include <catch2/catch_test_macros.hpp>

#include "plume/ParticleIDGen.h"


TEST_CASE("Testing ID generator class")
{
  // 1st ptr to the generator
  ParticleIDGen *id_gen;
  id_gen = ParticleIDGen::getInstance();

  uint32_t test;

  // test the generator
  test = id_gen->get();
  REQUIRE(test == 0);
  test = id_gen->get();
  REQUIRE(test == 1);
  test = id_gen->get();
  REQUIRE(test == 2);

  // 2nd ptr to the generator
  ParticleIDGen *id_gen_2;
  id_gen_2 = ParticleIDGen::getInstance();

  // test that the same id generator provide the same ids
  test = id_gen_2->get();
  REQUIRE(test == 3);

  // test the generator for vector
  std::vector<uint32_t> test_vec;
  test_vec.resize(10, 0);
  id_gen_2->get(test_vec);
  REQUIRE(test_vec[0] == 4);
  REQUIRE(test_vec[9] == 13);
}
