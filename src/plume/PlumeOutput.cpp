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
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file PlumeOutput.cpp
 * @brief This class handles saving output files for Eulerian binned Lagrangian particle data,
 * where this class handles the binning of the Lagrangian particle data
 * This is a specialized output class derived and inheriting from QESNetCDFOutput.
 *
 * @note child of QESNetCDFOutput
 * @sa QESNetCDFOutput
 */

#include "PlumeOutput.h"
#include "PLUMEGeneralData.h"

// note that this sets the output file and the bool for whether to do output, in the netcdf inherited classes
// in this case, output should always be done, so the bool for whether to do output is set to true
PlumeOutput::PlumeOutput(const PlumeInputData *PID, PLUMEGeneralData *PGD, std::string output_file)
  : QESNetCDFOutput(output_file)
{

  std::cout << "[PlumeOutput]\t Setting NetCDF file: " << output_file << std::endl;

  // setup copy of plume pointer so output data can be grabbed directly
  m_PGD = PGD;

  // setup output frequency control information
  averagingStartTime = m_PGD->getSimTimeStart() + PID->colParams->averagingStartTime;
  averagingPeriod = PID->colParams->averagingPeriod;

#if 0 
  // !!! Because collection parameters could not know anything about simulation duration at parse time,
  //  need to make this check now
  // Make sure the averagingStartTime is not greater than the simulation end time
  if (averagingStartTime > PID->plumeParams->simDur) {
    std::cerr << "[PlumeOutput] ERROR "
              << "(CollectionParameters checked during PlumeOutput) "
              << "input averagingStartTime must be smaller than or equal to the input simulation duration!" << std::endl;
    std::cerr << " averagingStartTime = \"" << averagingStartTime << "\", simDur = \"" << PID->plumeParams->simDur << "\"" << std::endl;
    exit(EXIT_FAILURE);
  }

  // !!! Because collection parameters could not know anything
  //  about the simulation duration at parse time, need to make this check now
  // Make sure averagingPeriod is not bigger than the simulation duration
  // LA note: averagingPeriod can be as big as the collection duration, or even smaller than the collection duration
  //  IF averagingPeriod is at least the same size or smaller than the simulation duration
  if (averagingPeriod > PID->plumeParams->simDur) {
    std::cerr << "[PlumeOutput] ERROR "
              << "(CollectionParameters checked during PlumeOutput): "
              << "input averagingPeriod must be smaller than or equal to the input simulation duration!" << std::endl;
    std::cerr << " averagingPeriod = \"" << averagingPeriod << "\", simDur = \"" << PID->plumeParams->simDur << "\"" << std::endl;
    exit(EXIT_FAILURE);
  }


  // Determine whether averagingStartTime needs adjusted to make the time average duration divide evenly by the averaging frequency
  // This is essentially always keeping the timeAvgEnd at what it is (end of the simulation), and adjusting the averagingStartTime
  // and outputStartTime to avoid slicing off an averaging and output time unless we have to.
  float avgDur = timeAvgEnd - averagingStartTime;
  // if doesn't divide evenly, need to adjust averagingStartTime
  // can determine if it divides evenly by comparing the quotient with the decimal division result
  //  if the values do not match, the division has a remainder
  //  here's hoping numerical error doesn't play a role
  float quotient = std::floor(avgDur / averagingPeriod);
  float decDivResult = avgDur / averagingPeriod;
  if (quotient != decDivResult) {
    // clever algorythm that always gets the exact number of time averages (and outputs)
    // when the time averaging duration divides evenly by the time averaging frequency
    // and rounds the number of time averages down to what it would be if the start time
    // were the next smallest evenly dividing number
    int nAvgs = std::floor(avgDur / averagingPeriod);

    // clever algorythm to always calculate the desired averaging start time based off the number of time averages
    // the averagingStartTime if not adjusting nAvgs
    float current_averagingStartTime = timeAvgEnd - averagingPeriod * (nAvgs);
    // the averagingStartTime if adjusting nAvgs. Note nAvgs has one extra averaging period
    float adjusted_averagingStartTime = timeAvgEnd - averagingPeriod * (nAvgs + 1);
    if (adjusted_averagingStartTime >= simStartTime) {
      // need to adjust the averagingStartTime to be the adjustedTimeAvgStart
      // warn the user that the averagingStartTime is being adjusted before adjusting averagingStartTime
      std::cout << "[PlumeOutput] "
                << "adjusting averagingStartTime because time averaging duration did not divide evenly by averagingPeriod" << std::endl;
      std::cout << "  original averagingStartTime = \"" << averagingStartTime
                << "\", timeAvgEnd = \"" << timeAvgEnd
                << "\", averagingPeriod = \"" << averagingPeriod
                << "\", new averagingStartTime = \"" << adjusted_averagingStartTime << "\"" << std::endl;
      averagingStartTime = adjusted_averagingStartTime;
    } else {
      // need to adjust the averagingStartTime to be the currentTimeAvgStart
      // warn the user that the averagingStartTime is being adjusted before adjusting averagingStartTime
      std::cout << "[PlumeOutput] "
                << "adjusting averagingStartTime because time averaging duration did not divide evenly by averagingPeriod" << std::endl;
      std::cout << "  original averagingStartTime = \"" << averagingStartTime
                << "\", timeAvgEnd = \"" << timeAvgEnd
                << "\", averagingPeriod = \"" << averagingPeriod
                << "\", new averagingStartTime = \"" << current_averagingStartTime << "\"" << std::endl;
      averagingStartTime = current_averagingStartTime;
    }
  }// else does divide evenly, no need to adjust anything so no else
#endif

  // set the initial next output time value
  nextOutputTime = averagingStartTime + averagingPeriod;

  /*
  // need the simulation timeStep for use in concentration averaging
  timeStep = PID->plumeParams->timeStep;


  // --------------------------------------------------------
  // setup information: sampling box/concentration
  // --------------------------------------------------------

  // Sampling box variables for calculating concentration data
  nBoxesX = PID->colParams->nBoxesX;
  nBoxesY = PID->colParams->nBoxesY;
  nBoxesZ = PID->colParams->nBoxesZ;

  lBndx = PID->colParams->boxBoundsX1;
  uBndx = PID->colParams->boxBoundsX2;
  lBndy = PID->colParams->boxBoundsY1;
  uBndy = PID->colParams->boxBoundsY2;
  lBndz = PID->colParams->boxBoundsZ1;
  uBndz = PID->colParams->boxBoundsZ2;

  boxSizeX = (uBndx - lBndx) / (nBoxesX);
  boxSizeY = (uBndy - lBndy) / (nBoxesY);
  boxSizeZ = (uBndz - lBndz) / (nBoxesZ);

  volume = boxSizeX * boxSizeY * boxSizeZ;

  // output concentration storage variables
  xBoxCen.resize(nBoxesX);
  yBoxCen.resize(nBoxesY);
  zBoxCen.resize(nBoxesZ);
  int zR = 0, yR = 0, xR = 0;
  for (int k = 0; k < nBoxesZ; ++k) {
    zBoxCen.at(k) = lBndz + (zR * boxSizeZ) + (boxSizeZ / 2.0);
    zR++;
  }
  for (int j = 0; j < nBoxesY; ++j) {
    yBoxCen.at(j) = lBndy + (yR * boxSizeY) + (boxSizeY / 2.0);
    yR++;
  }
  for (int i = 0; i < nBoxesX; ++i) {
    xBoxCen.at(i) = lBndx + (xR * boxSizeX) + (boxSizeX / 2.0);
    xR++;
  }

  // initialization of the container
  pBox.resize(nBoxesX * nBoxesY * nBoxesZ, 0);
  conc.resize(nBoxesX * nBoxesY * nBoxesZ, 0.0);
   */
  
  // --------------------------------------------------------
  // setup information:
  // --------------------------------------------------------

  /* moved to depostion class
  int nbrFace = WGD->wall_below_indices.size()
                + WGD->wall_above_indices.size()
                + WGD->wall_back_indices.size()
                + WGD->wall_front_indices.size()
                + WGD->wall_left_indices.size()
                + WGD->wall_right_indices.size();
  */

  // --------------------------------------------------------
  // setup the netcdf output information storage
  // --------------------------------------------------------

  setStartTime(m_PGD->getSimTimeStart());

  for (const auto &pm : m_PGD->models) {
    pm.second->stats->setOutput(this);
  }
  // setup desired output fields string
  // output_fields = { "x", "y", "z", "pBox", "conc", "tAvg" };
  // output_fields = { "x", "y", "z", "pBox", "conc", "tAvg", "xDep", "yDep", "zDep", "depcvol" };

  // set data dimensions, which in this case are cell-centered dimensions
  // space dimensions
  // NcDim NcDim_x = addDimension("x", nBoxesX);
  // NcDim NcDim_y = addDimension("y", nBoxesY);
  // NcDim NcDim_z = addDimension("z", nBoxesZ);


  // create attributes for time dimension
  // std::vector<NcDim> dim_vect_t;
  // dim_vect_t.push_back(NcDim_t);
  // createAttScalar("tAvg", "Averaging time", "s", dim_vect_t, &ongoingAveragingTime);
  /*
  createField("tAvg", "Averaging time", "s", "t", &ongoingAveragingTime);

  createDimension("x", "x-center collection box", "m", &xBoxCen);
  createDimension("y", "y-center collection box", "m", &yBoxCen);
  createDimension("z", "z-center collection box", "m", &zBoxCen);

  createDimensionSet("concentration", { "t", "z", "y", "x" });

  createField("pBox", "number of particle per box", "#ofPar", "concentration", &pBox);
  createField("conc", "concentration", "g m-3", "concentration", &conc);
*/
  /*
  // create attributes space dimensions
  std::vector<NcDim> dim_vect_x;
  dim_vect_x.push_back(NcDim_x);
  createField("x", "x-center collection box", "m", dim_vect_x, &xBoxCen);
  std::vector<NcDim> dim_vect_y;
  dim_vect_y.push_back(NcDim_y);
  createField("y", "y-center collection box", "m", dim_vect_y, &yBoxCen);
  std::vector<NcDim> dim_vect_z;
  dim_vect_z.push_back(NcDim_z);
  createField("z", "z-center collection box", "m", dim_vect_z, &zBoxCen);

  // create 3D vector and put in the dimensions (nt,nz,ny,nx).
  // !!! make sure the order is specificall nt,nz,ny,nx in this spot,
  //  the order doesn't seem to matter for other spots
  std::vector<NcDim> dim_vect_3d;
  dim_vect_3d.push_back(NcDim_t);
  dim_vect_3d.push_back(NcDim_z);
  dim_vect_3d.push_back(NcDim_y);
  dim_vect_3d.push_back(NcDim_x);


  // create attributes for all output information
  createField("pBox", "number of particle per box", "#ofPar", dim_vect_3d, &pBox);
  createField("conc", "concentration", "g m-3", dim_vect_3d, &conc);
*/
  // face dimensions
  // NcDim NcDim_nFace = addDimension("nFace", m_plume->deposition->nbrFace);
  // NcDim NcDim_x = addDimension("x",nBoxesX);
  // NcDim NcDim_y = addDimension("y",nBoxesY);
  // NcDim NcDim_z = addDimension("z",nBoxesZ);

  // NcDim NcDim_xDep = addDimension("xDep", m_PGD->deposition->x.size());
  // NcDim NcDim_yDep = addDimension("yDep", m_PGD->deposition->y.size());
  // NcDim NcDim_zDep = addDimension("zDep", m_PGD->deposition->z.size());

  /*
    std::vector<NcDim> dim_vect_xDep;
    dim_vect_xDep.push_back(NcDim_xDep);
    createField("xDep", "x-distance, deposition grid", "m", dim_vect_xDep, &(m_plume->deposition->x));
    std::vector<NcDim> dim_vect_yDep;
    dim_vect_yDep.push_back(NcDim_yDep);
    createField("yDep", "y-distance, deposition grid", "m", dim_vect_yDep, &(m_plume->deposition->y));
    std::vector<NcDim> dim_vect_zDep;
    dim_vect_zDep.push_back(NcDim_zDep);
    createField("zDep", "z-distance, deposition grid", "m", dim_vect_zDep, &(m_plume->deposition->z));

    std::vector<NcDim> dim_vectDep;
    dim_vectDep.push_back(NcDim_t);
    dim_vectDep.push_back(NcDim_zDep);
    dim_vectDep.push_back(NcDim_yDep);
    dim_vectDep.push_back(NcDim_xDep);
  */
  // createField("depcvol", "deposited mass", "g", dim_vectDep, &(m_plume->deposition->depcvol));

  // create attributes space dimensions
  // std::vector<NcDim> dim_vect_face;
  // dim_vect_face.push_back(NcDim_nFace);
  // createField("xface","x-face","m",dim_vect_face,&xBoxCen);
  // createField("yface","y-face","m",dim_vect_face,&xBoxCen);
  // createField("zface","z-face","m",dim_vect_face,&xBoxCen);

  // !!! make sure the order is specificall nt,nz,ny,nx in this spot,
  //  the order doesn't seem to matter for other spots
  // dim_vect_face.clear();
  // dim_vect_face.push_back(NcDim_t);
  // dim_vect_face.push_back(NcDim_nFace);

  // create output fields
  addOutputFields(set_all_output_fields);
}

// Save output at cell-centered values
void PlumeOutput::save(QEStime timeIn)
{
  if (timeIn > averagingStartTime) {

    // output to NetCDF file
    if (timeIn >= nextOutputTime) {
      // set output time for correct netcdf output
      timeCurrent = timeIn;

      // save the fields to NetCDF files
      saveOutputFields();

      // update the next output time value
      // so averaging and output only happens at the averaging frequency
      nextOutputTime = nextOutputTime + averagingPeriod;
    }
  }
};

void PlumeOutput::boxCount()
{
  /*
    // for all particles see where they are relative to the concentration collection boxes
    for (auto &par : *m_PGD->particles->tracer) {
      // because particles all start out as active now, need to also check the release time
      if (par.isActive) {

        // Calculate which collection box this particle is currently in.
        // The method is the same as the setInterp3Dindexing() function in the Eulerian class:
        //  Correct the particle position by the bounding box starting edge
        //  then divide by the dx of the boxes plus a small number, running a floor function on the result
        //  to get the index of the nearest concentration box node in the negative direction.
        //  No need to calculate the fractional distance between nearest nodes since not interpolating.
        // Because the particle position is offset by the bounding box starting edge,
        //  particles in a spot to the left of the box will have a negative index
        //  and particles in a spot to the right of the box will have an index greater than the number of boxes.
        // Because dividing is not just the box size, but is the box size plus a really small number,
        //  particles are considered in a box if they are on the left hand node to the right hand node
        //  so particles go outside the box if their indices are at nx-2, not nx-1.

        // x-direction
        int idx = floor((par.xPos - lBndx) / (boxSizeX + 1e-9));
        // y-direction
        int idy = floor((par.yPos - lBndy) / (boxSizeY + 1e-9));
        // z-direction
        int idz = floor((par.zPos - lBndz) / (boxSizeZ + 1e-9));

        // now, does the particle land in one of the boxes?
        // if so, add one particle to that box count
        if (idx >= 0 && idx <= nBoxesX - 1 && idy >= 0 && idy <= nBoxesY - 1 && idz >= 0 && idz <= nBoxesZ - 1) {
          int id = idz * nBoxesY * nBoxesX + idy * nBoxesX + idx;
          pBox[id]++;
          conc[id] = conc[id] + par.m * par.wdecay * timeStep;
        }

      }// is active == true

    }// particle loop

    for (auto &par : *m_PGD->particles->heavy) {
      // because particles all start out as active now, need to also check the release time
      if (par.isActive) {
        // x-direction
        int idx = floor((par.xPos - lBndx) / (boxSizeX + 1e-9));
        // y-direction
        int idy = floor((par.yPos - lBndy) / (boxSizeY + 1e-9));
        // z-direction
        int idz = floor((par.zPos - lBndz) / (boxSizeZ + 1e-9));

        // now, does the particle land in one of the boxes?
        if (idx >= 0 && idx <= nBoxesX - 1 && idy >= 0 && idy <= nBoxesY - 1 && idz >= 0 && idz <= nBoxesZ - 1) {
          int id = idz * nBoxesY * nBoxesX + idy * nBoxesX + idx;
          pBox[id]++;
          conc[id] = conc[id] + par.m * par.wdecay * timeStep;
        }
      }// is active == true
    }// particle loop
    */
}
