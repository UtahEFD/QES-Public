//
// Created by Fabien Margairaz on 4/2/23.
//
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <cstdlib>
#include <algorithm>

#include "util/calcTime.h"

#include "plume/PlumeInputData.hpp"
#include "util/NetCDFInput.h"
#include "util/QESout.h"

#include "test_WINDSGeneralData.h"

#include "winds/WINDSGeneralData.h"
#include "plume/Plume.hpp"
#include "plume/PLUMEGeneralData.h"
#include "plume/TracerParticle_Statistics.h"

#include "util/QESNetCDFOutput.h"
#include "plume/PlumeOutput.h"
#include "plume/PlumeOutputParticleData.h"

// float calcEntropy(int nbrBins, Plume *plume);
// void calcRMSE_wFluct(int nbrBins, Plume *plume, std::map<std::string, float> &rmse);
// void calcRMSE_delta_wFluct(int nbrBins, Plume *plume, double delta_t, std::map<std::string, float> &rmse);
float calcRMSE(std::vector<float> &A, std::vector<float> &B);
float calcMaxAbsErr(std::vector<float> &A, std::vector<float> &B);

TEST_CASE("Regression test of QES-Plume: uniform flow gaussian plume model")
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
        TGD->CoEps[cellID] = C0 * pow(uStar, 3)
                             / (0.4 * WGD->z[k]) * pow(1 - 0.85 * WGD->z[k] / zi, 3.0 / 2.0);
      }
    }
  }
  TGD->divergenceStress();

  // Create instance of Plume model class
  // auto *plume = new Plume(PID, WGD, TGD);
  auto *PGD = new PLUMEGeneralData(PID, WGD, TGD);

  // create output instance
  std::vector<QESNetCDFOutput *> outputVec;
  std::string outFile = QES_DIR;
  outFile.append("/scratch/UniformFlow_xDir_ContRelease_plumeOut.nc");

  // auto *plumeOutput = new PlumeOutput(PID, plume, outFile);

  // outputVec.push_back(plumeOutput);
  //  outputVec.push_back(new PlumeOutputParticleData(PID, plume,
  //  "../testCases/Sinusoidal3D/QES-data/Sinusoidal3D_0.01_12_particleInfo.nc"));

  // Run plume advection model
  QEStime endtime = WGD->timestamp[0] + PID->plumeParams->simDur;
  // plume->run(endtime, WGD, TGD, outputVec);
  PGD->run(endtime, WGD, TGD, outputVec);

  // compute run time information and print the elapsed execution time
  std::cout << "[QES-Plume] \t Finished." << std::endl;

  std::cout << "End run particle summary \n";
  PGD->showCurrentStatus();
  timers.printStoredTime("QES-Plume total runtime");
  std::cout << "##############################################################" << std::endl
            << std::endl;

  auto *test_model = dynamic_cast<TracerParticle_Model *>(PGD->models[PID->particleParams->particles.at(0)->tag]);
  TracerParticle_Concentration *test_conc = test_model->stats->concentration;
  // source info (hard coded because no way to access the source info here)
  float xS = 20;
  float yS = 50;
  float zS = 70;
  float Q = 400;// #par/s (source strength)
  float tRelease = 2100;// total time of release
  float Ntot = Q * tRelease;// total number of particles

  float CNorm = (uMean * H * H / Q);

  float dt = PID->plumeParams->timeStep;
  float tAvg = test_conc->averagingPeriod;

  // normalization of particle count #particle -> time-averaged # particle/m3
  float CC = dt / tAvg / test_conc->volume;

  // model variances
  float sigV = 1.78f * uStar;
  float sigW = powf(1.0f / (3.0f * 0.4f * 0.4f), 1.0f / 3.0f) * uStar;

  // working variables to calculate statistics
  float SSres = 0.0, SStot = 0.0;
  std::vector<float> xoH, maxRelErr, relRMSE;

  for (int i = 7; i < test_conc->nBoxesX; ++i) {
    // x-location of the profile
    float x = test_conc->xBoxCen[i] - xS;

    // time of flight of the particle
    float t = x / uMean;

    // horizontal and vertical spread rate
    float Ti = powf(2.5f * uStar / zi, -1);
    float To = 1.001;
    float Fy = powf(1.0f + powf(t / Ti, 0.5), -1);
    // float Fz=(1+0.945*(t/To)^0.8)^(-1);
    float Fz = Fy;
    float sigY = sigV * t * Fy;
    float sigZ = sigW * t * Fz;

    // calculate the normalize concentration from Gaussian plume model
    std::vector<float> CStarModel;
    CStarModel.resize(test_conc->nBoxesY * test_conc->nBoxesZ, 0.0);
    for (int k = 0; k < test_conc->nBoxesZ; ++k) {
      for (int j = 0; j < test_conc->nBoxesY; ++j) {
        float y = test_conc->yBoxCen[j] - yS;
        float z = test_conc->zBoxCen[k] - zS;
        CStarModel[j + k * test_conc->nBoxesY] = Q / (2 * M_PI * uMean * sigY * sigZ)
                                                 * expf(-0.5f * powf(y, 2) / powf(sigY, 2))
                                                 * expf(-0.5f * powf(z, 2) / powf(sigZ, 2))
                                                 * CNorm;
      }
    }

    // calculate the normalize concentration from QES-plume
    std::vector<float> CStarQES;
    CStarQES.resize(test_conc->nBoxesY * test_conc->nBoxesZ, 0.0);
    for (int k = 0; k < test_conc->nBoxesZ; ++k) {
      for (int j = 0; j < test_conc->nBoxesY; ++j) {
        int id = i + j * test_conc->nBoxesX + k * test_conc->nBoxesX * test_conc->nBoxesY;
        CStarQES[j + k * test_conc->nBoxesY] = (float)test_conc->pBox[id] * CC * CNorm;
      }
    }

    auto maxModel = *max_element(std::begin(CStarModel), std::end(CStarModel));
    // auto maxQES = *max_element(std::begin(CStarQES), std::end(CStarQES));

    // root mean square error
    float RMSE = calcRMSE(CStarQES, CStarModel);

    // maximum relative error
    float maxAbsErr = calcMaxAbsErr(CStarQES, CStarModel);

    xoH.push_back(x / H);
    maxRelErr.push_back(maxAbsErr / maxModel);
    relRMSE.push_back(RMSE / maxModel);

    // calculate r2
    float CStarQESMean = 0.0;
    for (auto k = 0u; k < CStarQES.size(); ++k) {
      CStarQESMean += CStarQES[k];
    }
    CStarQESMean /= (float)CStarQES.size();

    for (auto k = 0u; k < CStarQES.size(); ++k) {
      SSres += powf(CStarModel[k] - CStarQES[k], 2);
      SStot += powf(CStarQES[k] - CStarQESMean, 2);
    }
  }
  float R2 = 1.0f - SSres / SStot;

  // check the results
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << "TEST RESULTS" << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << "x / H \t maxRelErr \t relRMSE" << std::endl;
  for (auto k = 0u; k < xoH.size(); ++k) {
    printf("%.3f \t %.6f \t %.6f\n", xoH[k], maxRelErr[k], relRMSE[k]);
    // std::cout << xoH[k] << " " << maxRelErr[k] << "\t " << relRMSE[k] << std::endl;
  }
  std::cout << std::endl;
  std::cout << "QES-Plume r2 coeff: r2 = " << R2 << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;


  // Catch2 requirements
  REQUIRE(PGD->getNumRogueParticles() == 0);
  for (auto k = 0u; k < xoH.size(); ++k) {
    REQUIRE(maxRelErr[k] < 0.1);
    REQUIRE(relRMSE[k] < 0.02);
  }
  REQUIRE(R2 > 0.99);
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

float calcMaxAbsErr(std::vector<float> &A, std::vector<float> &B)
{
  int nbrBins = (int)A.size();
  vector<float> tmp(nbrBins);
  for (int k = 0; k < nbrBins; ++k) {
    tmp[k] = std::abs(A[k] - B[k]);
  }
  return *max_element(std::begin(tmp), std::end(tmp));
}
