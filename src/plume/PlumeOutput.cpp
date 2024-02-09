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
  setStartTime(m_PGD->getSimTimeStart());
  averagingStartTime = m_PGD->getSimTimeStart() + PID->colParams->averagingStartTime;
  averagingPeriod = PID->colParams->averagingPeriod;
  nextOutputTime = averagingStartTime + averagingPeriod;

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

  // --------------------------------------------------------
  // setup information:
  // --------------------------------------------------------
  for (const auto &pm : m_PGD->models) {
    pm.second->stats->setOutput(this);
  }

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
