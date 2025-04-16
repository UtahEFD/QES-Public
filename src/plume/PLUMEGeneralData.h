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

/** @file PLUMEGeneralData.hpp
 * @brief
 */

#pragma once

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <map>
#include <list>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "util/QEStime.h"
#include "util/calcTime.h"
#include "util/Vector3Float.h"
#include "util/VectorMath.h"
// #include "Matrix3.h"
#include "Random.h"
#include "RandomSingleton.h"

#include "util/QESNetCDFOutput.h"
#include "ParticleOutput.h"

#include "PLUMEInputData.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "Interp.h"
#include "InterpNearestCell.h"
#include "InterpPowerLaw.h"
#include "InterpTriLinear.h"

#include "GLE_Solver.h"

#include "DomainBoundaryConditions.h"

#include "Concentration.h"
#include "Deposition.h"

#include "WallReflection.h"
#include "WallReflection_StairStep.h"
#include "WallReflection_TriMesh.h"

#include "Particle.h"
#include "Source.h"

#include "ParticleModel.h"

class PlumeParameters
{
public:
  PlumeParameters() = default;
  PlumeParameters(std::string s1, bool b1, bool b2)
    : outputFileBasename(std::move(s1)), plumeOutput(b1), particleOutput(b2)
  {}
  ~PlumeParameters() = default;
  std::string outputFileBasename;
  bool plumeOutput = true;
  bool particleOutput = false;
};

class PLUMEGeneralData
{
  // define output class as friend
  friend class PlumeOutput;

public:
  PLUMEGeneralData(const PlumeParameters &, WINDSGeneralData *, TURBGeneralData *);
  // constructor
  // first makes a copy of the urb grid number of values and the domain size as determined by dispersion
  // then sets up the concentration sampling box information for output
  // next copies important input time values and calculates needed time information
  // lastly sets up the boundary condition functions and checks to make sure input BC's are valid
  PLUMEGeneralData(const PlumeParameters &, PlumeInputData *, WINDSGeneralData *, TURBGeneralData *);

  virtual ~PLUMEGeneralData();

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
  void run(QEStime, WINDSGeneralData *, TURBGeneralData *);

  // accessors:
  int getTotalParsToRelease() const { return totalParsToRelease; }
  int getNumReleasedParticles() const { return isReleasedCount; }
  int getNumRogueParticles() const { return isRogueCount; }
  int getNumCurrentParticles() const { return isActiveCount; }
  // int getNumNotActiveParticles() const { return isNotActiveCount; }
  // int getNumCurrentParticles() const { return particles->get_nbr_active(); }
  QEStime getSimTimeStart() const { return simTimeStart; }
  QEStime getSimTimeCurrent() const { return simTimeCurr; }

  void printProgress(const double &);
  void showCurrentStatus();

  void addParticleModel(ParticleModel *);

  std::map<std::string, ParticleModel *> models;

  GLE_Solver *GLE_solver;

#ifdef _OPENMP
  // if using openmp the RNG is not thread safe, use an array of RNG (one per thread)
  std::vector<Random *> threadRNG;
#else
  RandomSingleton *RNG = nullptr;
#endif

  PlumeParameters plumeParameters;
  // interpolation methods
  Interp *interp = nullptr;
  // wall reflection method
  WallReflection *wallReflect = nullptr;

  ParticleOutput *particleOutput = nullptr;


private:
  // temporary fix: adding copy of PID here:
  PlumeInputData *m_PID;

  // these values are calculated from the urb and turb grids by dispersion
  // they are used for applying boundary conditions at the walls of the domain
  float domainXstart = 0.0;// the domain starting x value, a copy of the value found by dispersion
  float domainXend = 0.0;// the domain ending x value, a copy of the value found by dispersion
  float domainYstart = 0.0;// the domain starting y value, a copy of the value found by dispersion
  float domainYend = 0.0;// the domain ending y value, a copy of the value found by dispersion
  float domainZstart = 0.0;// the domain starting z value, a copy of the value found by dispersion
  float domainZend = 0.0;// the domain ending z value, a copy of the value found by dispersion

  // protected:
  //  QES grid information
  // copies of the wind grid nx, ny, nz value
  int nx{};
  int ny{};
  int nz{};// a copy of the wind grid nz value
  // a copy of the wind grid dx, dy, dz value
  float dx{};
  float dy{};
  float dz{};
  float dxy{};
  float boxSizeZ{};

  // domain boundary conditions method
  DomainBC *domainBC_x = nullptr;
  DomainBC *domainBC_y = nullptr;
  DomainBC *domainBC_z = nullptr;

  // time variables
  float sim_dt = 0.0;// the simulation timestep
  QEStime simTimeStart;
  QEStime simTimeCurr;
  int simTimeIdx = 0;

  // some overall metadata for the set of particles
  int isRogueCount = 0;// just a total number of rogue particles per time iteration
  int isActiveCount = 0;
  int isReleasedCount = 0;

public:
  // important time variables not copied from dispersion
  // the Courant number, used to know how to divide up the simulation timestep into smaller per particle timesteps.
  float CourantNum = 0.0;
  float vel_threshold = 0.0;

  // this is the total number of particles expected to be released during the simulation
  // !!! this has to be calculated carefully inside the getInputSources() function
  int totalParsToRelease = 0;

  // tolerance used to determine whether makeRealizeable should be run on the stress tensor for a particle
  float invarianceTol = 1.0e-10;

  // used to know how frequently to print out information during the time loop of the solver
  float updateFrequency_timeLoop = 0.0;
  // used to know how frequently to print out information during the particle loop of the solver
  int updateFrequency_particleLoop = 0;

  // timer class useful for debugging and timing different operations
  calcTime timers;

  bool debug = false;
  bool verbose = false;

  // private:

public:
  void updateCounts();

  void applyBC(vec3 &, vec3 &, ParticleState &);

  // initialize
  void initializeParticleValues(const vec3 &pos, ParticleLSDM &particle_ldsm, TURBGeneralData *TGD);
  // this function gets sources from input data and adds them to the allSources vector
  // this function also calls the many check and calc functions for all the input sources
  // !!! note that these check and calc functions have to be called here
  //  because each source requires extra data not found in the individual source data
  // !!! totalParsToRelease needs calculated very carefully here using information from each of the sources
  // void getInputSources(PlumeInputData *);


  static float getMaxVariance(const TURBGeneralData *);

  // function for calculating the individual particle timestep from the courant number, the current velocity fluctuations,
  // and the grid size. Forces particles to always move only at one timestep at at time.
  // Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number
  float calcCourantTimestep(const float &u,
                            const float &v,
                            const float &w,
                            const float &timeRemainder);
  float calcCourantTimestep(const float &d,
                            const vec3 &vel,
                            const float &timeRemainder);

  // a function used at constructor time to set the pointer function to the desired BC type
  void setBCfunctions(const std::string &,
                      const std::string &,
                      const std::string &);

  PLUMEGeneralData() = default;
};
