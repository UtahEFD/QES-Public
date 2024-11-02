//
// Created by Fabien Margairaz on 11/19/23.
//
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <map>

#include "util/calcTime.h"

#include "winds/WINDSGeneralData.h"

#include "plume/PlumeInputData.hpp"
#include "plume/PLUMEGeneralData.h"

#include "plume/TracerParticle_Model.h"
#include "plume/HeavyParticle_Model.h"


TEST_CASE("Plume test inputs for multiple particle models")
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

  std::cout << "--------------------" << std::endl;
  // std::vector<ParticleModel *> test;
  std::map<std::string, ParticleModel *> test;

  for (auto p : PID->particleParams->particles) {

    std::cout << p->tag << std::endl;
    std::cout << p->particleType << std::endl;
    std::cout << p->sources.size() << std::endl;

    test[p->tag] = p->create();
    /*switch (p->particleType) {
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
    }*/
  }

  std::cout << "--------------------" << std::endl;

  /* Need C++17
   * for (const auto &[key, pm] : test) {
    std::cout << key << " " << pm->tag << std::endl;
    std::cout << pm->getParticleType() << std::endl;
  }*/

  for (const auto &pm : test) {
    std::cout << pm.first << " " << pm.second->tag << std::endl;
    std::cout << pm.second->getParticleType() << std::endl;
  }

  /*for (auto pm : test) {
    std::cout << pm->generateParticleList() << std::endl;
    std::cout << pm->getParticleType() << std::endl;
  }*/
}
TEST_CASE("Plume test inputs for multiple particle models with uniform winds and turbulence")
{
  // set up timer information for the simulation runtime
  calcTime timers;
  timers.startNewTimer("QES-Plume total runtime");// start recording execution time

  // parse command line arguments
  // PlumeArgs arguments;
  // arguments.processArguments(argc, argv);

  std::string qesPlumeParamFile = QES_DIR;
  // qesPlumeParamFile.append("/tests/regressionTests/plume_uniform_parameters.xml");
  qesPlumeParamFile.append("/tests/unitTests/plume_input_parameters.xml");

  // parse xml settings
  auto PID = new PlumeInputData(qesPlumeParamFile);

  float uMean = 2.0;
  float uStar = 0.174;
  float H = 70;
  float C0 = 5.7;
  float zi = 1000;

  int gridSize[3] = { 102, 102, 141 };
  float gridRes[3] = { 1.0, 1.0, 1.0 };
  qes::Domain domain(gridSize[0], gridSize[1], gridSize[2], gridRes[0], gridRes[1], gridRes[2]);
  WINDSGeneralData *WGD = new WINDSGeneralData(domain);
  WGD->timestamp.emplace_back("2020-01-01T00:00:00");
  TURBGeneralData *TGD = new TURBGeneralData(WGD);

  for (int k = 0; k < WGD->domain.nz(); ++k) {
    for (int j = 0; j < WGD->domain.ny(); ++j) {
      for (int i = 0; i < WGD->domain.nx(); ++i) {
        // int faceID = i + j * (WGD->nx) + k * (WGD->nx) * (WGD->ny);
        WGD->u[WGD->domain.face(i, j, k)] = uMean;
      }
    }
  }

  for (int k = 1; k < WGD->domain.nz() - 1; ++k) {
    for (int j = 0; j < WGD->domain.ny() - 1; ++j) {
      for (int i = 0; i < WGD->domain.nx() - 1; ++i) {
        // int cellID = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        int cellID = WGD->domain.cell(i, j, k);
        TGD->txx[cellID] = pow(2.50 * uStar, 2) * pow(1 - WGD->domain.z[k] / zi, 3. / 2.);
        TGD->tyy[cellID] = pow(1.78 * uStar, 2) * pow(1 - WGD->domain.z[k] / zi, 3. / 2.);
        TGD->tzz[cellID] = pow(1.27 * uStar, 2) * pow(1 - WGD->domain.z[k] / zi, 3. / 2.);
        TGD->txz[cellID] = -pow(uStar, 2) * pow(1 - WGD->domain.z[k] / zi, 3. / 2.);

        TGD->tke[cellID] = pow(uStar / 0.55, 2.0);
        TGD->CoEps[cellID] = C0 * pow(uStar, 3) / (0.4 * WGD->domain.z[k]) * pow(1 - 0.85 * WGD->domain.z[k] / zi, 3.0 / 2.0);
      }
    }
  }
  TGD->divergenceStress();

  // Create instance of Plume model class
  PlumeParameters PP("plume_input", false, false);
  auto *PGD = new PLUMEGeneralData(PP, PID, WGD, TGD);

  // Run plume advection model
  QEStime endtime = WGD->timestamp[0] + PID->plumeParams->simDur;
  PGD->run(endtime, WGD, TGD);

  // compute run time information and print the elapsed execution time
  std::cout << "[QES-Plume] \t Finished." << std::endl;
  PGD->showCurrentStatus();
  timers.printStoredTime("QES-Plume total runtime");
  std::cout << "##############################################################" << std::endl
            << std::endl;

  delete WGD;
  delete TGD;

  delete PID;
  delete PGD;
}