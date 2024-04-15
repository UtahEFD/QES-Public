/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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

/** @file NetCDFOutput.h */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <netcdf>

using namespace netCDF;
using namespace netCDF::exceptions;

/**
 * @class NetCDFOutput
 * @brief Handles the saving of output files.
 */
class NetCDFOutput
{
protected:
  NetCDFOutput() : outfile(nullptr) {}

  NcFile *outfile; /**< File to write. */
  std::map<std::string, NcVar> fields; /**< :document this: */

  std::string filename;

public:
  // initializer
  explicit NetCDFOutput(const std::string &);
  virtual ~NetCDFOutput();

  // setter
  NcDim addDimension(const std::string &, int size = 0);
  NcDim getDimension(const std::string &);
  void addField(const std::string &, const std::string &, const std::string &, std::vector<NcDim>, NcType);
  void addAtt(const std::string &, const std::string &, const std::string &);

  // save functions for 1D array (save 1D time)
  void saveField1D(const std::string &, const std::vector<size_t> &, int *);
  void saveField1D(const std::string &, const std::vector<size_t> &, float *);
  void saveField1D(const std::string &, const std::vector<size_t> &, double *);

  // save functions for 2D array (save 1D array, eg: x,y,z )
  void saveField2D(const std::string &, std::vector<int> &);
  void saveField2D(const std::string &, std::vector<float> &);
  void saveField2D(const std::string &, std::vector<double> &);

  // save functions for *D
  void saveField2D(const std::string &, const std::vector<size_t> &, const std::vector<size_t> &, std::vector<int> &);
  void saveField2D(const std::string &, const std::vector<size_t> &, const std::vector<size_t> &, std::vector<float> &);
  void saveField2D(const std::string &, const std::vector<size_t> &, const std::vector<size_t> &, std::vector<double> &);
  void saveField2D(const std::string &, const std::vector<size_t> &, const std::vector<size_t> &, std::vector<char> &);
};
