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
 * @file LocalMixing.cpp
 * @brief :document this:
 */

#include "LocalMixing.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

void LocalMixing::saveMixingLength(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  // open NetCDF file (constructor)
  mixLengthOut = new NetCDFOutput(WID->turbParams->filename);

  // create NcDimension for x,y,z (with ghost cell)
  NcDim NcDim_x = mixLengthOut->addDimension("x", WGD->domain.nx() - 1);
  NcDim NcDim_y = mixLengthOut->addDimension("y", WGD->domain.ny() - 1);
  NcDim NcDim_z = mixLengthOut->addDimension("z", WGD->domain.nz() - 1);

  std::vector<size_t> vector_index;
  std::vector<size_t> vector_size;

  vector_index = { 0, 0, 0 };
  vector_size = { static_cast<unsigned long>(WGD->domain.nz() - 1),
                  static_cast<unsigned long>(WGD->domain.ny() - 1),
                  static_cast<unsigned long>(WGD->domain.nx() - 1) };

  // create NetCDF filed in file
  mixLengthOut->addField(WID->turbParams->varname, "m", "distance to nearest object", { NcDim_z, NcDim_y, NcDim_x }, ncFloat);

  // dump mixingLengths to file
  mixLengthOut->saveField2D(WID->turbParams->varname, vector_index, vector_size, WGD->mixingLengths);
}
