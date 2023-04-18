//
// Created by Fabien Margairaz on 4/2/23.
//
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <cstdlib>
#include <algorithm>

#include "util/calcTime.h"


#include "plume/handlePlumeArgs.hpp"
#include "plume/PlumeInputData.hpp"
#include "util/NetCDFInput.h"
#include "util/QESout.h"

#include "../unitTests/test_WINDSGeneralData.h"

#include "winds/WINDSGeneralData.h"
#include "plume/Plume.hpp"

#include "util/QESNetCDFOutput.h"
#include "plume/PlumeOutput.h"
#include "plume/PlumeOutputParticleData.h"

// float calcEntropy(int nbrBins, Plume *plume);
// void calcRMSE_wFluct(int nbrBins, Plume *plume, std::map<std::string, float> &rmse);
// void calcRMSE_delta_wFluct(int nbrBins, Plume *plume, double delta_t, std::map<std::string, float> &rmse);
float calcRMSE(std::vector<float> &A, std::vector<float> &B);

TEST_CASE("Regression test of QES-Plume: sinusoidal stress")
{
  // set up timer information for the simulation runtime
  calcTime timers;
  timers.startNewTimer("QES-Plume total runtime");// start recording execution time

  // parse command line arguments
  // PlumeArgs arguments;
  // arguments.processArguments(argc, argv);

  std::string qesPlumeParamFile = QES_DIR;
  qesPlumeParamFile.append("/tests/regressionTests/plume_uniform_parameters.xml");

  // parse xml settings
  auto PID = new PlumeInputData(qesPlumeParamFile);

  float uMean = 2.0;
  float uStar = 0.174;
  float H = 70;
  float C0 = 5.7;
  float zi = 1000;

  int gridSize[3] = { 102, 102, 141 };
  float gridRes[3] = { 1.0, 1.0, 1.0 };

  WINDSGeneralData *WGD = new test_WINDSGeneralData(gridSize, gridRes);
  TURBGeneralData *TGD = new TURBGeneralData(WGD);

  for (int k = 0; k < WGD->nz; ++k) {
    for (int j = 0; j < WGD->ny; ++j) {
      for (int i = 0; i < WGD->nx; ++i) {
        int faceID = i + j * (WGD->nx) + k * (WGD->nx) * (WGD->ny);
        WGD->u[faceID] = uMean;
      }
    }
  }

  for (int k = 1; k < WGD->nz - 1; ++k) {
    for (int j = 0; j < WGD->ny - 1; ++j) {
      for (int i = 0; i < WGD->nx - 1; ++i) {
        int cellID = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        TGD->txx[cellID] = pow(2.50 * uStar, 2) * pow(1 - WGD->z[k] / zi, 3. / 2.);
        TGD->tyy[cellID] = pow(1.78 * uStar, 2) * pow(1 - WGD->z[k] / zi, 3. / 2.);
        TGD->tzz[cellID] = pow(1.27 * uStar, 2) * pow(1 - WGD->z[k] / zi, 3. / 2.);
        TGD->txz[cellID] = -pow(uStar, 2) * pow(1 - WGD->z[k] / zi, 3. / 2.);

        TGD->tke[cellID] = pow(uStar / 0.55, 2.0);
        TGD->CoEps[cellID] = C0 * pow(uStar, 3) / (0.4 * WGD->z[k]) * pow(1 - 0.85 * WGD->z[k] / zi, 3.0 / 2.0);
      }
    }
  }
  TGD->divergenceStress();

  // Create instance of Plume model class
  auto *plume = new Plume(PID, WGD, TGD);

  // create output instance
  std::vector<QESNetCDFOutput *> outputVec;
  // always supposed to output lagrToEulOutput data
  std::string outFile = QES_DIR;
  outFile.append("/testCases/UniformFlow_ContRelease/QES-data/UniformFlow_xDir_ContRelease_plumeOut.nc");

  outputVec.push_back(new PlumeOutput(PID, plume, outFile));
  // outputVec.push_back(new PlumeOutputParticleData(PID, plume, "../testCases/Sinusoidal3D/QES-data/Sinusoidal3D_0.01_12_particleInfo.nc"));

  // Run plume advection model
  QEStime endtime = WGD->timestamp[0] + PID->plumeParams->simDur;
  plume->run(endtime, WGD, TGD, outputVec);

  // compute run time information and print the elapsed execution time
  std::cout << "[QES-Plume] \t Finished." << std::endl;

  std::cout << "End run particle summary \n";
  // plume->showCurrentStatus();
  timers.printStoredTime("QES-Plume total runtime");
  std::cout << "##############################################################" << std::endl
            << std::endl;


  REQUIRE(plume->getNumRogueParticles() == 0);
  /*
  // check the results
  // float entropy = calcEntropy(25, plume);

  // std::map<std::string, float> rmse;
  // calcRMSE_wFluct(20, plume, rmse);
  // calcRMSE_delta_wFluct(20, plume, PID->plumeParams->timeStep, rmse);

  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << "TEST RESULTS" << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << "Entropy: S = " << entropy << std::endl;
  std::cout << "Nbr Rogue Particle: " << plume->getNumRogueParticles() << std::endl;
  std::cout << "RMSE on mean of wFluct: " << rmse["mean_wFluct"] << std::endl;
  std::cout << "RMSE on variance of wFluct: " << rmse["mean_wFluct"] << std::endl;
  std::cout << "RMSE on mean of wFluct/delta t: " << rmse["mean_delta_wFluct"] << std::endl;
  std::cout << "RMSE on var of wFluct/delta t: " << rmse["var_delta_wFluct"] << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;

  REQUIRE(plume->getNumRogueParticles() == 0);
  REQUIRE(entropy > -0.04);
  REQUIRE(rmse["mean_wFluct"] < 0.03);
  REQUIRE(rmse["var_wFluct"] < 0.03);
  REQUIRE(rmse["mean_delta_wFluct"] < 0.6);
  REQUIRE(rmse["var_delta_wFluct"] < 0.3);
   */
}

float calcRMSE(std::vector<float> &A, std::vector<float> &B)
{
  int nbrBins = (int)A.size();
  float rmse_var = 0.0;
  for (int k = 0; k < nbrBins; ++k) {
    rmse_var += pow(A[k] - B[k], 2);
  }
  return sqrt(rmse_var / (float)nbrBins);
}
