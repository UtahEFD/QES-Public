/****************************************************************************
 * Copyright (c) 2026 University of La Rochelle and TIPEE
 *
 * This file is part of QES / pyQES.
 *
 * GPL-3.0 License
 ****************************************************************************/

/** @file run_plume.cpp */

#include "run_plume.h"

#include <stdexcept>

#include "util/QEStime.h"

#include "plume/PLUMEInputData.h"
#include "plume/PLUMEGeneralData.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

namespace pyqes {

PlumeRunResult run_plume(const std::string &plumeXmlPath,
                         const std::string &windsFile,
                         const std::string &turbFile,
                         const std::string &outBasename,
                         bool particleOutput)
{
  if (plumeXmlPath.empty())
    throw std::runtime_error("run_plume: empty QES-Plume XML path");
  if (windsFile.empty())
    throw std::runtime_error("run_plume: empty winds workspace file path");
  if (turbFile.empty())
    throw std::runtime_error("run_plume: empty turbulence file path");

  PlumeInputData *PID = new PlumeInputData(plumeXmlPath);
  if (!PID)
    throw std::runtime_error("run_plume: cannot read QES-Plume input file: " + plumeXmlPath);

  WINDSGeneralData *WGD = new WINDSGeneralData(windsFile);
  TURBGeneralData *TGD = new TURBGeneralData(turbFile, WGD);

  PlumeParameters plumeParameters(outBasename, true, particleOutput);
  PLUMEGeneralData *PGD = new PLUMEGeneralData(plumeParameters, PID, WGD, TGD);

  for (int index = 0; index < WGD->totalTimeIncrements; index++) {
    WGD->loadNetCDFData(index);
    TGD->loadNetCDFData(index);

    QEStime endTime = WGD->nextTimeInstance(index, PID->plumeParams->simDur);
    PGD->run(endTime, WGD, TGD);
  }

  PGD->showCurrentStatus();

  PlumeRunResult result;
  result.plumeOut = outBasename + "_plumeOut.nc";
  if (particleOutput)
    result.particleOut = outBasename + "_particleInfo.nc";

  delete WGD;
  delete TGD;
  delete PID;
  delete PGD;

  return result;
}

}// namespace pyqes
