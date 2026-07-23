/****************************************************************************
 * Copyright (c) 2026 University of La Rochelle and TIPEE
 *
 * This file is part of QES / pyQES.
 *
 * GPL-3.0 License
 ****************************************************************************/

/** @file run_plume.h
 * @brief Reusable QES-Plume workflow runner used by the pyQES bindings.
 *
 * Factors the body of qesPlume/qesPlumeMain.cpp into a callable function.
 */

#pragma once

#include <string>

namespace pyqes {

/**
 * @struct PlumeRunResult
 * @brief Output NetCDF files produced by a QES-Plume run.
 */
struct PlumeRunResult
{
  std::string plumeOut; /**< Concentration output file (*_plumeOut.nc) */
  std::string particleOut; /**< Optional Lagrangian particle info (*_particleInfo.nc) */
};

/**
 * Run the QES-Plume advection model on precomputed wind/turb fields.
 *
 * @param plumeXmlPath   Path to the QES-Plume XML parameter file.
 * @param windsFile      Path to the QES-Winds workspace NetCDF (*_windsWk.nc).
 * @param turbFile       Path to the QES-Turb NetCDF (*_turbOut.nc).
 * @param outBasename    Basename for the produced NetCDF files.
 * @param particleOutput Also write the debug Lagrangian particle file.
 * @return Paths of the files that were written.
 * @throws std::runtime_error on invalid inputs.
 */
PlumeRunResult run_plume(const std::string &plumeXmlPath,
                         const std::string &windsFile,
                         const std::string &turbFile,
                         const std::string &outBasename,
                         bool particleOutput);

}// namespace pyqes
