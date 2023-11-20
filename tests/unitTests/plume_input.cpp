//
// Created by Fabien Margairaz on 11/19/23.
//
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <cstdlib>
#include <algorithm>

#include "util/calcTime.h"

#include "plume/PlumeInputData.hpp"

TEST_CASE("plume input test")
{
  // set up timer information for the simulation runtime
  calcTime timers;
  timers.startNewTimer("QES-Plume total runtime");// start recording execution time

  // parse command line arguments
  // PlumeArgs arguments;
  // arguments.processArguments(argc, argv);

  std::string qesPlumeParamFile = QES_DIR;
  qesPlumeParamFile.append("/tests/unitTests/plume_input_parameters.xml");

  // parse xml settings
  auto PID = new PlumeInputData(qesPlumeParamFile);
  std::cout << PID->particleParams->particles.size() << std::endl;
  std::cout << PID->particleParams->particles[0]->particleType << std::endl;
  std::cout << PID->particleParams->particles[0]->sources.size() << std::endl;
}