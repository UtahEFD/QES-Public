/****************************************************************************
 * Copyright (c) 2026 University of La Rochelle and TIPEE
 *
 * This file is part of QES / pyQES.
 *
 * GPL-3.0 License
 ****************************************************************************/

/** @file run_winds.cpp */

#include "run_winds.h"

#include <stdexcept>
#include <vector>

#include "util/QESNetCDFOutput.h"

#include "qes/Domain.h"

#include "winds/WINDSInputData.h"
#include "winds/WINDSGeneralData.h"
#include "winds/WINDSOutputVisualization.h"
#include "winds/WINDSOutputWorkspace.h"
#include "winds/TURBGeneralData.h"
#include "winds/TURBOutput.h"
#include "winds/Solver.h"
#include "winds/SolverFactory.h"

namespace pyqes {

WindsRunResult run_winds(const std::string &xmlPath,
                         int solveType,
                         const std::string &outBasename,
                         bool visuOutput,
                         bool wkspOutput,
                         bool turbOutput)
{
  if (xmlPath.empty())
    throw std::runtime_error("run_winds: empty XML path");

  // Parse the base QES XML file (simulation parameters).
  WINDSInputData *WID = new WINDSInputData(xmlPath);
  if (!WID)
    throw std::runtime_error("run_winds: cannot read QES input file: " + xmlPath);

  if (turbOutput && !WID->turbParams) {
    delete WID;
    throw std::runtime_error("run_winds: turbulence requested but <turbParams> missing in " + xmlPath);
  }

  qes::Domain domain(WID->simParams->domain, WID->simParams->grid);

  WINDSGeneralData *WGD = new WINDSGeneralData(WID, domain, solveType);

  WindsRunResult result;
  std::vector<QESNetCDFOutput *> outputVec;
  if (visuOutput) {
    result.windsOut = outBasename + "_windsOut.nc";
    outputVec.push_back(new WINDSOutputVisualization(WGD, WID, result.windsOut));
  }
  if (wkspOutput) {
    result.windsWk = outBasename + "_windsWk.nc";
    outputVec.push_back(new WINDSOutputWorkspace(WGD, result.windsWk));
  }

  TURBGeneralData *TGD = nullptr;
  if (turbOutput) {
    TGD = new TURBGeneralData(WID, WGD);
    result.turbOut = outBasename + "_turbOut.nc";
    outputVec.push_back(new TURBOutput(TGD, result.turbOut));
  }

  SolverFactory solverFactory;
  Solver *solver = solverFactory.create(solveType, WGD->domain, WID->simParams->tolerance);

  const int tempMaxIter = WID->simParams->maxIterations;

  for (int index = 0; index < WGD->totalTimeIncrements; index++) {
    if (!WID->simParams->wrfCoupling)
      WGD->printTimeProgress(index);

    WGD->resetICellFlag();
    WGD->applyWindProfile(WID, index, solveType);
    WGD->applyParametrizations(WID);

    solver->resetLambda();

    if (WID->simParams->logLawFlag == 1) {
      solver->solve(WGD, tempMaxIter);
      WGD->u0 = WGD->u;
      WGD->v0 = WGD->v;
      WGD->w0 = WGD->w;
      WGD->wall->wallLogBC(WGD, true);
      WGD->u = WGD->u0;
      WGD->v = WGD->v0;
      WGD->w = WGD->w0;
    } else {
      solver->solve(WGD, tempMaxIter);
    }

    if (TGD != nullptr)
      TGD->run();

    for (auto &id_out : outputVec)
      id_out->save(WGD->timestamp[index]);
  }

  delete solver;
  delete WID;
  delete WGD;
  delete TGD;
  for (auto p : outputVec)
    delete p;

  return result;
}

}// namespace pyqes
