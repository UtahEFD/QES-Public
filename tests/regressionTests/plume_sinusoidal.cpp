//
// Created by Fabien Margairaz on 3/24/23.
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

#include "plume/PLUMEGeneralData.h"

#include "util/QESNetCDFOutput.h"
#include "plume/PlumeOutput.h"
#include "plume/PlumeOutputParticleData.h"

float calcEntropy(int nbrBins, ManagedContainer<TracerParticle> *particles);
void calcRMSE_wFluct(int nbrBins, ManagedContainer<TracerParticle> *particles, std::map<std::string, float> &rmse);
void calcRMSE_delta_wFluct(int nbrBins, ManagedContainer<TracerParticle> *particles, double delta_t, std::map<std::string, float> &rmse);
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
  qesPlumeParamFile.append("/tests/regressionTests/plume_sinusoidal_parameters.xml");

  // parse xml settings
  auto PID = new PlumeInputData(qesPlumeParamFile);

  int gridSize[3] = { 20, 20, 50 };
  float gridRes[3] = { 1.0 / 20.0, 1.0 / 20.0, 1.0 / 49.0 };

  WINDSGeneralData *WGD = new test_WINDSGeneralData(gridSize, gridRes);
  TURBGeneralData *TGD = new TURBGeneralData(WGD);

  std::vector<float> sig2_new(WGD->nz - 1);

  for (int k = 0; k < WGD->nz - 1; ++k) {
    sig2_new[k] = 1.1 + sin(2.0 * M_PI * WGD->z[k]);
  }

  for (int k = 0; k < WGD->nz - 1; ++k) {
    for (int j = 0; j < WGD->ny - 1; ++j) {
      for (int i = 0; i < WGD->nx - 1; ++i) {
        int cellID = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        TGD->txx[cellID] = sig2_new[k];
        TGD->tyy[cellID] = sig2_new[k];
        TGD->tzz[cellID] = sig2_new[k];
        TGD->tke[cellID] = pow(WGD->z[k] * pow(sig2_new[k], 3.0 / 2.0), 2.0 / 3.0);
        TGD->CoEps[cellID] = 4.0 * pow(sig2_new[k], 3.0 / 2.0);
      }
    }
  }
  TGD->divergenceStress();

  // Create instance of Plume model class
  // auto *plume = new Plume(PID, WGD, TGD);
  PlumeParameters PP("", false, false);
  auto *PGD = new PLUMEGeneralData(PP, PID, WGD, TGD);
  // create output instance
  // std::vector<QESNetCDFOutput *> outputVec;
  // always supposed to output lagrToEulOutput data
  // outputVec.push_back(new PlumeOutput(PID, plume, "../testCases/Sinusoidal3D/QES-data/Sinusoidal3D_0.01_12_plumeOut.nc"));
  // outputVec.push_back(new PlumeOutputParticleData(PID, plume, "../testCases/Sinusoidal3D/QES-data/Sinusoidal3D_0.01_12_particleInfo.nc"));

  // Run plume advection model
  QEStime endtime = WGD->timestamp[0] + PID->plumeParams->simDur;
  PGD->run(endtime, WGD, TGD);

  // compute run time information and print the elapsed execution time
  std::cout << "[QES-Plume] \t Finished." << std::endl;

  std::cout << "End run particle summary \n";
  // plume->showCurrentStatus();
  timers.printStoredTime("QES-Plume total runtime");
  std::cout << "##############################################################" << std::endl
            << std::endl;

  TracerParticle_Model *test_model = dynamic_cast<TracerParticle_Model *>(PGD->models[PID->particleParams->particles.at(0)->tag]);

  // check the results
  float entropy = calcEntropy(25, test_model->get_particles());

  std::map<std::string, float> rmse;
  calcRMSE_wFluct(20, test_model->get_particles(), rmse);
  calcRMSE_delta_wFluct(20, test_model->get_particles(), PID->plumeParams->timeStep, rmse);

  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << "TEST RESULTS" << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << "Entropy: S = " << entropy << std::endl;
  std::cout << "Nbr Rogue Particle: " << PGD->getNumRogueParticles() << std::endl;
  std::cout << "RMSE on mean of wFluct: " << rmse["mean_wFluct"] << std::endl;
  std::cout << "RMSE on variance of wFluct: " << rmse["mean_wFluct"] << std::endl;
  std::cout << "RMSE on mean of wFluct/delta t: " << rmse["mean_delta_wFluct"] << std::endl;
  std::cout << "RMSE on var of wFluct/delta t: " << rmse["var_delta_wFluct"] << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;

  REQUIRE(PGD->getNumRogueParticles() == 0);
  REQUIRE(entropy > -0.04);
  REQUIRE(rmse["mean_wFluct"] < 0.03);
  REQUIRE(rmse["var_wFluct"] < 0.03);
  REQUIRE(rmse["mean_delta_wFluct"] < 0.6);
  REQUIRE(rmse["var_delta_wFluct"] < 0.3);
}

float calcEntropy(int nbrBins, ManagedContainer<TracerParticle> *particles)
{
  std::vector<float> pBin;
  pBin.resize(nbrBins, 0.0);
  /*for (auto &parItr : plume->particleList) {
    int k = floor(parItr->zPos / (1.0 / nbrBins + 1e-9));
    pBin[k]++;
  }*/
  for (auto &par : *particles) {
    int k = floor(par.zPos / (1.0 / nbrBins + 1e-9));
    pBin[k]++;
  }

  float expectedParsPerBin = particles->get_nbr_active() / (float)nbrBins;
  float entropy = 0.0;

  std::vector<float> proba;
  proba.resize(nbrBins, 0.0);
  for (int k = 0; k < nbrBins; ++k) {
    proba[k] = pBin[k] / expectedParsPerBin;
    entropy += proba[k] * log(proba[k]);
  }

  // for (int k = 1; k < nbrBins; ++k) {
  //   std::cout << proba[k] << std::endl;
  // }

  return -entropy;
}

void calcRMSE_wFluct(int nbrBins, ManagedContainer<TracerParticle> *particles, std::map<std::string, float> &rmse)
{
  std::vector<float> pBin;
  pBin.resize(nbrBins, 0.0);
  std::vector<float> pBin_mean;
  pBin_mean.resize(nbrBins, 0.0);

  // calculate mean of fluctuation
  /*for (auto &parItr : plume->particleList) {
    int k = floor(parItr->zPos / (1.0 / nbrBins + 1e-9));
    pBin[k]++;
    pBin_mean[k] += parItr->wFluct;
  }*/
  for (auto &par : *particles) {
    int k = floor(par.zPos / (1.0 / nbrBins + 1e-9));
    pBin[k]++;
    pBin_mean[k] += par.wFluct;
  }
  for (int k = 0; k < nbrBins; ++k) {
    pBin_mean[k] /= pBin[k];
  }

  // calculate theoretical mean of fluctuation
  std::vector<float> wFluct_mean;
  wFluct_mean.resize(nbrBins, 0.0);
  // std::cout << "RMSE on mean wFluct " << calcRMSE(pBin_mean, wFluct_mean) << std::endl;
  rmse["mean_wFluct"] = calcRMSE(pBin_mean, wFluct_mean);

  // calculate theoretical variance of fluctuation (stress)
  std::vector<float> pBin_var;
  pBin_var.resize(nbrBins, 0.0);
  /*for (auto &parItr : plume->particleList) {
    int k = floor(parItr->zPos / (1.0 / nbrBins + 1e-9));
    pBin_var[k] += pow(parItr->wFluct - pBin_mean[k], 2) / pBin[k];
  }*/
  for (auto &par : *particles) {
    int k = floor(par.zPos / (1.0 / nbrBins + 1e-9));
    pBin_var[k] += pow(par.wFluct - pBin_mean[k], 2) / pBin[k];
  }

  // calculate theoretical mean of time derivative of fluctuation
  std::vector<float> sig2(nbrBins);
  for (int k = 0; k < nbrBins; ++k) {
    sig2[k] = 1.1 + sin(2.0 * M_PI * (k + 0.5) * (1.0 / nbrBins));
  }
  // std::cout << "RMSE on var wFluct " << calcRMSE(pBin_var, sig2) << std::endl;
  rmse["var_wFluct"] = calcRMSE(pBin_var, sig2);
}

void calcRMSE_delta_wFluct(int nbrBins, ManagedContainer<TracerParticle> *particles, double delta_t, std::map<std::string, float> &rmse)
{
  std::vector<float> pBin;
  pBin.resize(nbrBins, 0.0);
  std::vector<float> pBin_mean;
  pBin_mean.resize(nbrBins, 0.0);

  // calculate mean of time derivative of fluctuation
  /*for (auto &parItr : plume->particleList) {
    int k = floor(parItr->zPos / (1.0 / nbrBins + 1e-9));
    pBin[k]++;
    pBin_mean[k] += parItr->delta_wFluct;
  }*/
  for (auto &par : *particles) {
    int k = floor(par.zPos / (1.0 / nbrBins + 1e-9));
    pBin[k]++;
    pBin_mean[k] += par.delta_wFluct;
  }

  for (int k = 0; k < nbrBins; ++k) {
    pBin_mean[k] /= pBin[k];
  }

  // calculate variance of time derivative of fluctuation
  std::vector<float> pBin_var;
  pBin_var.resize(nbrBins, 0.0);
  /*for (auto &parItr : plume->particleList) {
    int k = floor(parItr->zPos / (1.0 / nbrBins + 1e-9));
    pBin_var[k] += pow(parItr->delta_wFluct - pBin_mean[k], 2) / pBin[k];
  }*/
  for (auto &par : *particles) {
    int k = floor(par.zPos / (1.0 / nbrBins + 1e-9));
    pBin_var[k] += pow(par.delta_wFluct - pBin_mean[k], 2) / pBin[k];
  }
  for (int k = 0; k < nbrBins; ++k) {
    pBin_var[k] /= delta_t;
    pBin_mean[k] /= delta_t;
  }

  // calculate theoretical mean of time derivative of fluctuation
  std::vector<float> wFluct_mean(nbrBins);
  for (int k = 0; k < nbrBins; ++k) {
    wFluct_mean[k] = 2.0 * M_PI * cos(2.0 * M_PI * (k + 0.5) * (1.0 / nbrBins));
  }
  // std::cout << "RMSE on mean delta wFluct " << calcRMSE(pBin_mean, wFluct_mean) << std::endl;
  rmse["mean_delta_wFluct"] = calcRMSE(pBin_mean, wFluct_mean);

  // calculate theoretical variance of time derivative of fluctuation
  std::vector<float> CoEps(nbrBins);
  for (int k = 0; k < nbrBins; ++k) {
    CoEps[k] = 4.0 * pow(1.1 + sin(2.0 * M_PI * (k + 0.5) * (1.0 / nbrBins)), 3.0 / 2.0);
  }
  // std::cout << "RMSE on var delta wFluct " << calcRMSE(pBin_var, CoEps) << std::endl;
  rmse["var_delta_wFluct"] = calcRMSE(pBin_var, CoEps);
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
