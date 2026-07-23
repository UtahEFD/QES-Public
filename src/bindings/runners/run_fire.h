/****************************************************************************
 * Copyright (c) 2026 University of La Rochelle and TIPEE
 *
 * This file is part of QES / pyQES.
 *
 * GPL-3.0 License
 ****************************************************************************/

/** @file run_fire.h
 * @brief Reusable QES-Fire workflow runner used by the pyQES bindings.
 *
 * Factors the body of qesFire/qesFireMain.cpp into a callable function.
 */

#pragma once

#include <string>

namespace pyqes {

/**
 * @struct FireRunResult
 * @brief Output NetCDF files produced by a QES-Fire run.
 */
struct FireRunResult
{
  std::string fireOut; /**< Fire output file (*_fireOutput.nc) */
  std::string plumeOut; /**< Optional smoke concentration file (*_plumeOut.nc) */
};

/**
 * Run the coupled QES-Fire workflow.
 *
 * @param windsXmlPath  Path to the QES-Winds/Fire XML parameter file.
 * @param plumeXmlPath  Path to the QES-Plume XML (empty disables smoke plume).
 * @param solveType     Solver type (1 = CPU, 2/3/4 = GPU variants when built).
 * @param outBasename   Basename for the produced NetCDF files.
 * @param compTurb      Compute QES-Turb fields.
 * @param fireWindsOff  Disable fire-induced winds.
 * @return Paths of the files that were written.
 * @throws std::runtime_error on invalid inputs.
 */
FireRunResult run_fire(const std::string &windsXmlPath,
                       const std::string &plumeXmlPath,
                       int solveType,
                       const std::string &outBasename,
                       bool compTurb,
                       bool fireWindsOff);

}// namespace pyqes
