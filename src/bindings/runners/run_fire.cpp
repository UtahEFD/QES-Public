/****************************************************************************
 * Copyright (c) 2026 University of La Rochelle and TIPEE
 *
 * This file is part of QES / pyQES.
 *
 * GPL-3.0 License
 ****************************************************************************/

/** @file run_fire.cpp */

#include "run_fire.h"

#include <stdexcept>
#include <vector>

#include "util/QESNetCDFOutput.h"
#include "util/QEStime.h"

#include "qes/Domain.h"

#include "winds/WINDSInputData.h"
#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"
#include "winds/TURBOutput.h"
#include "winds/Solver.h"
#include "winds/SolverFactory.h"

#include "plume/PLUMEInputData.h"
#include "plume/PLUMEGeneralData.h"

#include "fire/Fire.h"
#include "fire/FIREOutput.h"
#include "fire/SourceFire.h"

namespace pyqes {

FireRunResult run_fire(const std::string &windsXmlPath,
                       const std::string &plumeXmlPath,
                       int solveType,
                       const std::string &outBasename,
                       bool compTurb,
                       bool fireWindsOff)
{
  if (windsXmlPath.empty())
    throw std::runtime_error("run_fire: empty QES-Winds XML path");

  const bool compPlume = !plumeXmlPath.empty();

  WINDSInputData *WID = new WINDSInputData(windsXmlPath);
  if (!WID)
    throw std::runtime_error("run_fire: cannot read QES input file: " + windsXmlPath);

  if (compTurb && !WID->turbParams) {
    delete WID;
    throw std::runtime_error("run_fire: turbulence requested but <turbParams> missing in " + windsXmlPath);
  }

  qes::Domain domain(WID->simParams->domain, WID->simParams->grid);
  WINDSGeneralData *WGD = new WINDSGeneralData(WID, domain, solveType);

  bool GPUFLAG = false;
#ifdef HAS_CUDA
  if (solveType == DYNAMIC_P || solveType == Global_M || solveType == Shared_M)
    GPUFLAG = true;
#endif

  Fire *fire = new Fire(WID, WGD, GPUFLAG);
  fire->FuelMap(WID, WGD);

  FireRunResult result;
  result.fireOut = outBasename + "_fireOutput.nc";

  std::vector<QESNetCDFOutput *> outFire;
  outFire.push_back(new FIREOutput(WGD, fire, result.fireOut));

  SolverFactory solverFactory;
  Solver *solver = solverFactory.create(solveType, WGD->domain, WID->simParams->tolerance);

  QEStime simTimeStart = WGD->timestamp[0];
  QEStime simTimeCurr = simTimeStart;

  std::vector<float> Fu0(domain.numFaceCentered(), 0.0);
  std::vector<float> Fv0(domain.numFaceCentered(), 0.0);
  std::vector<float> Fw0(domain.numFaceCentered(), 0.0);

  TURBGeneralData *TGD = nullptr;
  if (compTurb)
    TGD = new TURBGeneralData(WID, WGD);

  PlumeInputData *PID = nullptr;
  PLUMEGeneralData *PGD = nullptr;
  if (compPlume) {
    PID = new PlumeInputData(plumeXmlPath);
    if (!PID) {
      throw std::runtime_error("run_fire: cannot read QES-Plume input file: " + plumeXmlPath);
    }
    PGD = new PLUMEGeneralData(PlumeParameters(outBasename, true, false), PID, WGD, TGD);
    PGD->addParticleModel(new ParticleModel("smoke"));
    result.plumeOut = outBasename + "_plumeOut.nc";
  }

  for (int index = 0; index < WGD->totalTimeIncrements; index++) {
    WGD->resetICellFlag();
    WGD->applyWindProfile(WID, index, solveType);
    WGD->applyParametrizations(WID);
    solver->solve(WGD, WID->simParams->maxIterations);

    if (TGD != nullptr) TGD->run();

    Fu0 = WGD->u0;
    Fv0 = WGD->v0;
    Fw0 = WGD->w0;

    simTimeCurr = WGD->timestamp[index];
    QEStime endtime;
    if (WGD->totalTimeIncrements == 1)
      endtime = WGD->timestamp[index] + WID->fires->fireDur;
    else if (index == WGD->totalTimeIncrements - 1)
      endtime = simTimeStart + WID->fires->fireDur;
    else
      endtime = WGD->timestamp[index + 1];

    while (simTimeCurr < endtime) {
      WGD->u0 = Fu0;
      WGD->v0 = Fv0;
      WGD->w0 = Fw0;

      if (!fireWindsOff) {
        fire->LevelSetNB(WGD);
        fire->potential(WGD);
      }

      WGD->applyParametrizations(WID);
      solver->solve(WGD, WID->simParams->maxIterations);
      if (TGD != nullptr) TGD->run();

      fire->LevelSetNB(WGD);
      fire->move(WGD);

      if (PGD != nullptr) {
        for (int j = 1; j < domain.ny() - 2; j++) {
          for (int i = 1; i < domain.nx() - 2; i++) {
            int idx = i + j * (domain.nx() - 1);
            if (fire->smoke_flag[idx] == 1) {
              float x_pos = i * domain.dx();
              float y_pos = j * domain.dy();
              float z_pos = WGD->terrain[idx] + 1;
              int ppt = 20;
              FireSourceBuilder FSB;
              FSB.setSourceParam({ x_pos, y_pos, z_pos },
                                 simTimeCurr,
                                 simTimeCurr + fire->fire_cells[idx].properties.tau,
                                 ppt);
              PGD->models["smoke"]->addSource(FSB.create());
              fire->smoke_flag[idx] = 0;
            }
          }
        }
        QEStime pendtime = simTimeCurr + fire->dt;
        PGD->run(pendtime, WGD, TGD);
      }

      simTimeCurr += fire->dt;

      for (auto &out : outFire)
        out->save(simTimeCurr);
    }
  }

  delete solver;
  delete WID;
  delete WGD;
  delete TGD;
  delete PID;
  delete PGD;
  delete fire;
  for (auto p : outFire)
    delete p;

  return result;
}

}// namespace pyqes
