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

/** @file Plume.hpp 
 * @brief
 */

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <cmath>
#include <cstring>

#include "util/QEStime.h"
#include "util/calcTime.h"
#include "util/Vector3.h"
//#include "Matrix3.h"
#include "Random.h"

#include "util/QESNetCDFOutput.h"
#include "PlumeOutput.h"
#include "PlumeOutputParticleData.h"

#include "PlumeInputData.hpp"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "Interp.h"
#include "InterpNearestCell.h"
#include "InterpPowerLaw.h"
#include "InterpTriLinear.h"

#include "DomainBoundaryConditions.h"

#include "Deposition.h"

#include "WallReflection.h"
#include "WallReflection_StairStep.h"

#include "Particle.hpp"

#include "SourcePoint.hpp"
#include "SourceLine.hpp"
#include "SourceCircle.hpp"
#include "SourceCube.hpp"
#include "SourceFullDomain.hpp"

class Plume
{

public:
  Plume(WINDSGeneralData *, TURBGeneralData *);
  // constructor
  // first makes a copy of the urb grid number of values and the domain size as determined by dispersion
  // then sets up the concentration sampling box information for output
  // next copies important input time values and calculates needed time information
  // lastly sets up the boundary condition functions and checks to make sure input BC's are valid
  Plume(PlumeInputData *, WINDSGeneralData *, TURBGeneralData *);

  virtual ~Plume()
  {}

  // this is the plume solver. It performs a time integration of the particle positions and particle velocity fluctuations
  // with calculations done on a per particle basis. During each iteration, temporary single value particle information
  // is taken from the overall particle list to do the calculations, with an update to the overall list particle information
  // using the single value particle information at the end of the calculations.
  // Note that command line outputs are setup to not give all the information for each particle and timestep unless errors occur
  //  but are setup to be user controlled at input to limit the information to a sample of the particle information.
  // LA note: Outputting particle information for each particle for each timestep overwhelms the command line output file readers
  //  cause storing the command line output gets ridiculous.
  // LA future work: Need to add a CFL condition where the user specifies a courant number that varies from 0 to 1
  //  that is used to do an additional time remainder time integration loop for each particle, forcing particles to only
  //  move one cell at a time.
  void run(QEStime, WINDSGeneralData *, TURBGeneralData *, std::vector<QESNetCDFOutput *>);

  int getTotalParsToRelease() const { return totalParsToRelease; }// accessor

  int getNumReleasedParticles() const { return nParsReleased; }// accessor
  int getNumRogueParticles() const { return isRogueCount; }// accessor
  int getNumNotActiveParticles() const { return isNotActiveCount; }// accessor
  int getNumCurrentParticles() const { return particleList.size(); }// accessor

  QEStime getSimTimeStart() const { return simTimeStart; }
  QEStime getSimTimeCurrent() const { return simTimeCurr; }

  void showCurrentStatus();

  // This the storage for all particles
  // the sources can set these values, then the other values are set using urb and turb info using these values
  std::list<Particle *> particleList;

  Interp *interp;

  Deposition *deposition;

  // these values are calculated from the urb and turb grids by dispersion
  // they are used for applying boundary conditions at the walls of the domain
  double domainXstart;// the domain starting x value, a copy of the value found by dispersion
  double domainXend;// the domain ending x value, a copy of the value found by dispersion
  double domainYstart;// the domain starting y value, a copy of the value found by dispersion
  double domainYend;// the domain ending y value, a copy of the value found by dispersion
  double domainZstart;// the domain starting z value, a copy of the value found by dispersion
  double domainZend;// the domain ending z value, a copy of the value found by dispersion

protected:
  // QES grid information
  int nx;// a copy of the urb grid nx value
  int ny;// a copy of the urb grid ny value
  int nz;// a copy of the urb grid nz value
  double dx;// a copy of the urb grid dx value, eventually could become an array
  double dy;// a copy of the urb grid dy value, eventually could become an array
  double dz;// a copy of the urb grid dz value, eventually could become an array
  double dxy;//a copy of the urb grid dz value, eventually could become an array

  /*
  // these values are calculated from the urb and turb grids by dispersion
  // they are used for applying boundary conditions at the walls of the domain
  double domainXstart;// the domain starting x value, a copy of the value found by dispersion
  double domainXend;// the domain ending x value, a copy of the value found by dispersion
  double domainYstart;// the domain starting y value, a copy of the value found by dispersion
  double domainYend;// the domain ending y value, a copy of the value found by dispersion
  double domainZstart;// the domain starting z value, a copy of the value found by dispersion
  double domainZend;// the domain ending z value, a copy of the value found by dispersion
  */

  WallReflection *wallReflect;

  DomainBC *domainBC_x;
  DomainBC *domainBC_y;
  DomainBC *domainBC_z;

  // time variables
  double sim_dt;// the simulation timestep
  QEStime simTimeStart;
  QEStime simTimeCurr;
  int simTimeIdx;

  // some overall metadata for the set of particles
  int isRogueCount;// just a total number of rogue particles per time iteration
  int isNotActiveCount;// just a total number of inactive active particles per time iteration

  // important time variables not copied from dispersion
  double CourantNum;// the Courant number, used to know how to divide up the simulation timestep into smaller per particle timesteps. Copied from input
  double vel_threshold;

  // ALL Sources that will be used
  std::vector<SourceType *> allSources;
  // this is the global counter of particles released (used to set particleID)
  int nParsReleased;

  // this is the total number of particles expected to be released during the simulation
  // !!! this has to be calculated carefully inside the getInputSources() function
  int totalParsToRelease;

  double invarianceTol;// this is the tolerance used to determine whether makeRealizeable should be run on the stress tensor for a particle
  int updateFrequency_timeLoop;// used to know how frequently to print out information during the time loop of the solver
  int updateFrequency_particleLoop;// used to know how frequently to print out information during the particle loop of the solver

  // timer class useful for debugging and timing different operations
  calcTime timers;

  // copies of debug related information from the input arguments
  bool doParticleDataOutput;
  bool outputSimInfoFile;
  std::string outputFolder;
  std::string caseBaseName;
  bool debug;

  bool verbose;

  void setParticleVals(WINDSGeneralData *, TURBGeneralData *, std::list<Particle *>);
  // this function gets sources from input data and adds them to the allSources vector
  // this function also calls the many check and calc functions for all the input sources
  // !!! note that these check and calc functions have to be called here
  //  because each source requires extra data not found in the individual source data
  // !!! totalParsToRelease needs calculated very carefully here using information from each of the sources
  void getInputSources(PlumeInputData *);

  // this function generates the list of particle to be released at a given time
  int generateParticleList(float, WINDSGeneralData *, TURBGeneralData *);

  // this function scrubs the inactive particle for the particle list (particleList)
  void scrubParticleList();

  double getMaxVariance(const TURBGeneralData *);

  // this function moves (advects) one particle
  void advectParticle(double, std::list<Particle *>::iterator, WINDSGeneralData *, TURBGeneralData *);


  void depositParticle(double,
                       double,
                       double,
                       double,
                       double,
                       double,
                       double,
                       double,
                       double,
                       double,
                       double,
                       double,
                       double,
                       std::list<Particle *>::iterator,
                       WINDSGeneralData *,
                       TURBGeneralData *);

  // function for calculating the individual particle timestep from the courant number, the current velocity fluctuations,
  // and the grid size. Forces particles to always move only at one timestep at at time.
  // Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number
  double calcCourantTimestep(const double &u,
                             const double &v,
                             const double &w,
                             const double &timeRemainder);
  double calcCourantTimestep(const double &d,
                             const double &u,
                             const double &v,
                             const double &w,
                             const double &timeRemainder);

  // utility functions for the plume solver
  void calcInvariants(const double &txx,
                      const double &txy,
                      const double &txz,
                      const double &tyy,
                      const double &tyz,
                      const double &tzz,
                      double &invar_xx,
                      double &invar_yy,
                      double &invar_zz);

  void makeRealizable(double &txx,
                      double &txy,
                      double &txz,
                      double &tyy,
                      double &tyz,
                      double &tzz);

  bool invert3(double &A_11,
               double &A_12,
               double &A_13,
               double &A_21,
               double &A_22,
               double &A_23,
               double &A_31,
               double &A_32,
               double &A_33);

  void matmult(const double &A_11,
               const double &A_12,
               const double &A_13,
               const double &A_21,
               const double &A_22,
               const double &A_23,
               const double &A_31,
               const double &A_32,
               const double &A_33,
               const double &b_11,
               const double &b_21,
               const double &b_31,
               double &x_11,
               double &x_21,
               double &x_31);

  // a function used at constructor time to set the pointer function to the desired BC type
  void setBCfunctions(std::string xBCtype, std::string yBCtype, std::string zBCtype);


private:
  Plume();
};

inline void Plume::showCurrentStatus()
{
  std::cout << "----------------------------------------------------------------- \n";
  std::cout << "Current simulation time: " << simTimeCurr << "\n";
  std::cout << "Simulation run time: " << simTimeCurr - simTimeStart << "\n";
  std::cout << "Total number of particles released: " << nParsReleased << "\n";
  std::cout << "Current number of particles in simulation: " << particleList.size() << "\n";
  std::cout << "Number of rogue particles: " << isRogueCount << "\n";
  std::cout << "Number of deleted particles: " << isNotActiveCount << "\n";
  std::cout << "----------------------------------------------------------------- \n"
            << std::flush;
}