/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
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

/** @file util_output.cpp */

#include <iostream>
#include <list>
#include <string>

#include "util/QEStime.h"
#include "util/QESFileOutput_v2.h"
#include "util/DataSource.h"
#include "util/QESNetCDFOutput_v2.h"

class Concentration : public DataSource
{
public:
  Concentration()
  {
    int nBoxesX = 5;
    int nBoxesY = 5;
    int nBoxesZ = 2;


    // output concentration storage variables
    xBoxCen.resize(nBoxesX);
    yBoxCen.resize(nBoxesY);
    zBoxCen.resize(nBoxesZ);

    int zR = 0, yR = 0, xR = 0;
    for (int k = 0; k < nBoxesZ; ++k) {
      zBoxCen.at(k) = 0 + (zR * 1) + (1 / 2.0);
      zR++;
    }
    for (int j = 0; j < nBoxesY; ++j) {
      yBoxCen.at(j) = 0 + (yR * 1) + (1 / 2.0);
      yR++;
    }
    for (int i = 0; i < nBoxesX; ++i) {
      xBoxCen.at(i) = 0 + (xR * 1) + (1 / 2.0);
      xR++;
    }

    ongoingAveragingTime = 1000;

    // initialization of the container
    pBox.resize(nBoxesX * nBoxesY * nBoxesZ, 0);
    conc.resize(nBoxesX * nBoxesY * nBoxesZ, 0.0);
  }
  ~Concentration() = default;

  void prepareDataAndPushToFile(QEStime t) override
  {
    std::cout << "[Concentration] prepare data and push to file" << std::endl;
    pushToFile(t);
  }

protected:
  void setOutputFields() override
  {
    std::cout << "[Concentration] call set output" << std::endl;

    defineDimension("x_c", "x-center collection box", "m", &xBoxCen);
    defineDimension("y_c", "y-center collection box", "m", &yBoxCen);
    defineDimension("z_c", "z-center collection box", "m", &zBoxCen);

    defineDimensionSet("concentration", { "t", "z_c", "y_c", "x_c" });

    defineVariable("t_avg", "Averaging time", "s", "time", &ongoingAveragingTime);
    defineVariable("p", "number of particle per box", "#ofPar", "concentration", &pBox);
    defineVariable("c", "concentration", "g m-3", "concentration", &conc);
  }

  // output concentration storage variables
  float ongoingAveragingTime;
  std::vector<float> xBoxCen, yBoxCen, zBoxCen;// list of x,y, and z points for the concentration sampling box information
  std::vector<int> pBox;// sampling box particle counter (for average)
  std::vector<float> conc;// concentration values (for output)
};

class Spectra : public DataSource
{
public:
  Spectra()
  {
    int nBoxesX = 5;
    int nBoxesY = 5;

    // output concentration storage variables
    xBoxCen.resize(nBoxesX);
    yBoxCen.resize(nBoxesY);

    int zR = 0, yR = 0, xR = 0;
    for (int j = 0; j < nBoxesY; ++j) {
      yBoxCen.at(j) = 0 + (yR * 1) + (1 / 2.0);
      yR++;
    }
    for (int i = 0; i < nBoxesX; ++i) {
      xBoxCen.at(i) = 0 + (xR * 1) + (1 / 2.0);
      xR++;
    }

    ongoingAveragingTime = 0;

    // initialization of the container
    sp.resize(nBoxesX * nBoxesY, 0.0);
  }

  ~Spectra() = default;

  void prepareDataAndPushToFile(QEStime t) override
  {
    std::cout << "[Spectra] prepare data and push to file" << std::endl;
    ongoingAveragingTime += 1000;
    pushToFile(t);
  }

protected:
  void setOutputFields() override
  {
    std::cout << "[Spectra] call set output" << std::endl;

    defineDimension("k_x", "x-wavenumber", "m-1", &xBoxCen);
    defineDimension("k_y", "y-wavenumber", "m-1", &yBoxCen);

    defineDimensionSet("spectra", { "t", "k_y", "k_x" });

    defineVariable("t_collection", "Collecting time", "s", "time", &ongoingAveragingTime);
    defineVariable("s", "spectra", "m-3", "spectra", &sp);
  }

  // output concentration storage variables
  float ongoingAveragingTime;
  std::vector<float> xBoxCen, yBoxCen, zBoxCen;// list of x,y, and z points for the concentration sampling box information
  std::vector<float> sp;// concentration values (for output)
};

int main()
{
  // new file for output (set as NetCDF file)
  QESFileOutput_Interface *testFile = new QESNetCDFOutput_v2("test.nc");

  // new data source and attach to output file
  DataSource *concentration = new Concentration();
  testFile->attachDataSource(concentration);

  // new data source and attach to output file
  DataSource *spectra = new Spectra();
  testFile->attachDataSource(spectra);

  // set a new time ('now' from default constructor)
  QEStime t;
  // set the start time in the output file
  testFile->setStartTime(t);

  // increment time and add a new time entry in the file
  t += 120;
  testFile->newTimeEntry(t);

  // do so calculations and push to file
  concentration->prepareDataAndPushToFile(t);
  spectra->prepareDataAndPushToFile(t);

  // increment time and add a new time entry in the file
  t += 120;
  testFile->newTimeEntry(t);

  // alternatively, the file can call all the data source to push to file.
  // note: prepare data will NOT be called in that case, but data sources
  //       that already pushed to file will not push again
  testFile->save(t);
}
