#include <catch2/catch_test_macros.hpp>

#include "plume/IDGenerator.h"


TEST_CASE("Testing ID generator class")
{
  IDGenerator *id_gen;
  id_gen = IDGenerator::getInstance();

  uint32_t test;

  test = id_gen->get();
  REQUIRE(test == 0);
  test = id_gen->get();
  REQUIRE(test == 1);
  test = id_gen->get();
  REQUIRE(test == 2);


  IDGenerator *id_gen_2;
  id_gen_2 = IDGenerator::getInstance();

  test = id_gen_2->get();
  REQUIRE(test == 3);

  std::vector<uint32_t> test_vec;
  test_vec.resize(10, 0);

  id_gen_2->get(test_vec);
  REQUIRE(test_vec[0] == 4);
  REQUIRE(test_vec[9] == 13);
}
