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

/**
 * @file NetCDFOutput.cpp
 * @brief Handles the saving of output files.
 */

#include "NetCDFOutput.h"

#include <iostream>

using namespace netCDF;
using namespace netCDF::exceptions;

// constructor, linked to NetCDF file, replace mode only
NetCDFOutput ::NetCDFOutput(const std::string &output_file)
{
  std::cout << "[NetCDFOutput] \t Writing to " << output_file << std::endl;
  outfile = new NcFile(output_file, NcFile::replace);
}


NcDim NetCDFOutput ::addDimension(const std::string &name, int size)
{

  if (size) {
    return outfile->addDim(name, size);
  } else {
    return outfile->addDim(name);
  }
}

NcDim NetCDFOutput ::getDimension(const std::string &name)
{

  return outfile->getDim(name);
}

void NetCDFOutput ::addField(const std::string &name,
                             const std::string &units,
                             const std::string &long_name,
                             std::vector<NcDim> dims,
                             NcType type)
{

  NcVar var;

  var = outfile->addVar(name, type, dims);
  var.putAtt("units", units);
  var.putAtt("long_name", long_name);
  fields[name] = var;
}

void NetCDFOutput ::addAtt(const std::string &name, const std::string &att_name, const std::string &att_string)
{

  NcVar var = fields[name];

  var.putAtt(att_name, att_string);
}

// 1D -> int
void NetCDFOutput ::saveField1D(const std::string &name, const std::vector<size_t> &index, int *data)
{

  // write output data
  NcVar var = fields[name];
  var.putVar(index, data);
  outfile->sync();
}

// 1D -> float
void NetCDFOutput ::saveField1D(const std::string &name, const std::vector<size_t> &index, float *data)
{

  // write output data
  NcVar var = fields[name];
  var.putVar(index, data);
  outfile->sync();
}

// 1D -> double
void NetCDFOutput ::saveField1D(const std::string &name, const std::vector<size_t> &index, double *data)
{

  // write output data
  NcVar var = fields[name];
  var.putVar(index, data);
  outfile->sync();
}

// 2D -> int
void NetCDFOutput ::saveField2D(const std::string &name, std::vector<int> &data)
{

  // write output data
  NcVar var = fields[name];
  var.putVar(&data[0]);
  outfile->sync();
}

// 2D -> float
void NetCDFOutput ::saveField2D(const std::string &name, std::vector<float> &data)
{

  // write output data
  NcVar var = fields[name];
  var.putVar(&data[0]);
  outfile->sync();
}

// 2D -> double
void NetCDFOutput ::saveField2D(const std::string &name, std::vector<double> &data)
{

  // write output data
  NcVar var = fields[name];
  var.putVar(&data[0]);
  outfile->sync();
}

// *D -> int
void NetCDFOutput ::saveField2D(const std::string &name, const std::vector<size_t> &index, const std::vector<size_t> &size, std::vector<int> &data)
{

  // write output data
  NcVar var = fields[name];
  var.putVar(index, size, &data[0]);
  outfile->sync();
}

// *D -> float
void NetCDFOutput ::saveField2D(const std::string &name, const std::vector<size_t> &index, const std::vector<size_t> &size, std::vector<float> &data)
{

  // write output data
  NcVar var = fields[name];
  var.putVar(index, size, &data[0]);
  outfile->sync();
}

// *D -> double
void NetCDFOutput ::saveField2D(const std::string &name, const std::vector<size_t> &index, const std::vector<size_t> &size, std::vector<double> &data)
{

  // write output data
  NcVar var = fields[name];
  var.putVar(index, size, &data[0]);
  outfile->sync();
}

// *D -> char
void NetCDFOutput ::saveField2D(const std::string &name, const std::vector<size_t> &index, const std::vector<size_t> &size, std::vector<char> &data)
{

  // write output data
  NcVar var = fields[name];
  var.putVar(index, size, &data[0]);
  outfile->sync();
}
