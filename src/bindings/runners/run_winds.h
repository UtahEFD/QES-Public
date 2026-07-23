/****************************************************************************
 * Copyright (c) 2026 University of La Rochelle and TIPEE
 *
 * This file is part of QES / pyQES.
 *
 * GPL-3.0 License
 ****************************************************************************/

/** @file run_winds.h
 * @brief Reusable QES-Winds workflow runner used by the pyQES bindings.
 *
 * Factors the body of qesWinds/qesWindsMain.cpp into a callable function that
 * returns the produced output paths instead of terminating the process.
 */

#pragma once

#include <string>

namespace pyqes {

/**
 * @struct WindsRunResult
 * @brief Output NetCDF files produced by a QES-Winds run (empty when disabled).
 */
struct WindsRunResult
{
  std::string windsOut; /**< Cell-centered visualization file (*_windsOut.nc) */
  std::string windsWk; /**< Workspace file consumed by QES-Plume (*_windsWk.nc) */
  std::string turbOut; /**< Turbulence field file (*_turbOut.nc) */
};

/**
 * Run the QES-Winds solver on a QES XML input file.
 *
 * @param xmlPath      Path to the QES-Winds XML parameter file.
 * @param solveType    Solver type (1 = CPU, 2/3/4 = GPU variants when built).
 * @param outBasename  Basename for the produced NetCDF files.
 * @param visuOutput   Write the cell-centered visualization file.
 * @param wkspOutput   Write the workspace file (required by QES-Plume).
 * @param turbOutput   Compute QES-Turb and write the turbulence file.
 * @return Paths of the files that were written.
 * @throws std::runtime_error on invalid inputs or missing turbulence params.
 */
WindsRunResult run_winds(const std::string &xmlPath,
                         int solveType,
                         const std::string &outBasename,
                         bool visuOutput,
                         bool wkspOutput,
                         bool turbOutput);

}// namespace pyqes
