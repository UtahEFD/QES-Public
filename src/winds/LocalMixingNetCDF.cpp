/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file LocalMixingNetCDF.cpp
 * @brief :document this:
 * @sa LocalMixing
 */

#include "LocalMixingNetCDF.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

void LocalMixingNetCDF::defineMixingLength(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  // open NetCDF file (constructor)
  NetCDFInput *mixLengthInput;
  mixLengthInput = new NetCDFInput(WID->turbParams->filename);

  int nx_f, ny_f, nz_f;

  // nx,ny,ny from file
  mixLengthInput->getDimensionSize("x", nx_f);
  mixLengthInput->getDimensionSize("y", ny_f);
  mixLengthInput->getDimensionSize("z", nz_f);

  if (nx_f != WGD->domain.nx() - 1 || ny_f != WGD->domain.ny() - 1 || nz_f != WGD->domain.nz() - 1) {
    std::cout << "[ERROR] \t domain size error in " << WID->turbParams->filename << std::endl;
    exit(EXIT_FAILURE);
  }

  // access variable (to check if exist)
  NcVar NcVar_mixlength;
  mixLengthInput->getVariable(WID->turbParams->varname, NcVar_mixlength);

  if (!NcVar_mixlength.isNull()) {// => mixlength in NetCDF file
    // netCDF variables
    std::vector<size_t> start;
    std::vector<size_t> count;
    start = { 0, 0, 0 };
    count = { static_cast<unsigned long>(nz_f),
              static_cast<unsigned long>(ny_f),
              static_cast<unsigned long>(nx_f) };

    // read in mixilength
    mixLengthInput->getVariableData(WID->turbParams->varname, start, count, WGD->mixingLengths);
  } else {
    std::cout << "[ERROR] \t no field " << WID->turbParams->varname << " in "
              << WID->turbParams->filename << std::endl;
    exit(EXIT_FAILURE);
  }
}
