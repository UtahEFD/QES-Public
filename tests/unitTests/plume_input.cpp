//
// Created by Fabien Margairaz on 11/19/23.
//
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <cstdlib>
#include <algorithm>

#include "util/calcTime.h"

#include "plume/PlumeInputData.hpp"

#include "plume/TracerParticle_Model.h"
#include "plume/HeavyParticle_Model.h"

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


  std::vector<ParticleModel *> test;
  for (auto p : PID->particleParams->particles) {
    std::cout << p->tag << std::endl;
    switch (p->particleType) {
    case ParticleType::tracer: {
      test.emplace_back(new TracerParticle_Model(PID, dynamic_cast<PI_TracerParticle *>(p)));
      break;
    }
    case ParticleType::heavy: {
      test.emplace_back(new HeavyParticle_Model(PID, dynamic_cast<PI_HeavyParticle *>(p)));
      break;
    }
    default:
      exit(1);
    }
  }

  for (auto pm : test) {
    std::cout << pm->getParticleType() << std::endl;
  }
}