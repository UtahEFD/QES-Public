/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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
 * @file NetCDFInput.cpp
 * @brief Handles the reading of input files.
 *
 * @author Jeremy Gibbs
 * @author Fabien Margairaz
 */

#include <iostream>
#include "NetCDFInput.h"

using namespace netCDF;
using namespace netCDF::exceptions;

NetCDFInput ::NetCDFInput(std::string input_file)
{

  std::cout << "[NetCDFInput] \t Reading " << input_file << std::endl;
  infile = new NcFile(input_file, NcFile::read);
}

void NetCDFInput ::getDimension(std::string name, NcDim &external)
{

  external = infile->getDim(name);
}

void NetCDFInput ::getDimensionSize(std::string name, int &external)
{

  external = infile->getDim(name).getSize();
}

void NetCDFInput ::getVariable(std::string name, NcVar &external)
{

  external = infile->getVar(name);
}

// 1D -> int
void NetCDFInput ::getVariableData(std::string name, std::vector<int> &external)
{

  infile->getVar(name).getVar(&external[0]);
}
// 1D -> float
void NetCDFInput ::getVariableData(std::string name, std::vector<float> &external)
{

  infile->getVar(name).getVar(&external[0]);
}
// 1D -> double
void NetCDFInput ::getVariableData(std::string name, std::vector<double> &external)
{

  infile->getVar(name).getVar(&external[0]);
}

// *D -> int
void NetCDFInput ::getVariableData(std::string name, const std::vector<size_t> start, std::vector<size_t> count, std::vector<int> &external)
{

  infile->getVar(name).getVar(start, count, &external[0]);
}
// *D -> float
void NetCDFInput ::getVariableData(std::string name, const std::vector<size_t> start, std::vector<size_t> count, std::vector<float> &external)
{

  infile->getVar(name).getVar(start, count, &external[0]);
}
// *D -> double
void NetCDFInput ::getVariableData(std::string name, const std::vector<size_t> start, std::vector<size_t> count, std::vector<double> &external)
{

  infile->getVar(name).getVar(start, count, &external[0]);
}
