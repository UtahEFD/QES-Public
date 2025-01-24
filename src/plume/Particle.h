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

/** @file Particle.hpp
 * @brief This class represents information stored for each particle
 */

#pragma once

#include <cmath>
#include <utility>

#include "util/QEStime.h"
#include "util/VectorMath.h"
#include "util/ParseInterface.h"

enum ParticleType {
  base,
  tracer,
  heavy
};

enum ParticleState {
  ACTIVE,
  INACTIVE,
  ROGUE
};

typedef struct
{
  int length;

  ParticleState *state;
  uint32_t *ID;

  float *CoEps;
  float *nuT;

  vec3 *pos;

  vec3 *velMean;

  vec3 *velFluct;
  vec3 *velFluct_old;
  vec3 *delta_velFluct;

  mat3sym *tau;
  mat3sym *tau_old;

  vec3 *flux_div;

} particle_array;

struct particle_physcal
{
  float *d;
  float *m;
  float *rho;
};

class ParticleControl
{
public:
  ParticleControl() : state(INACTIVE) {}
  ~ParticleControl() = default;

  // state of the particle
  ParticleState state;

  void reset()
  {
    state = ACTIVE;
  }
};

// core particle properties
class ParticleCore
{
public:
  ParticleCore() = default;
  ~ParticleCore() = default;

  // id of particle (for tracking purposes)
  uint32_t ID{};
  // position for the particle
  vec3 pos{};
  // particle diameter diameter [microns]
  float d{};
  // particle mass [g]
  float m{};// particle mass [g]
  // density of particle g/m3
  float rho{};
  // decay (1 - fraction) particle decayed [0,1]
  float wdecay{};

  void reset(const uint32_t &ID0, const vec3 &p0)
  {
    ID = ID0;
    pos = p0;
    d = 0.0;
    m = 0.0;
    rho = 0.0;
    wdecay = 1.0;
  }

  void reset(const uint32_t &ID0, const vec3 &p0, const float &d0, const float &m0, const float &rho0)
  {
    ID = ID0;
    pos = p0;
    d = d0;
    m = m0;
    rho = rho0;
    wdecay = 1.0;
  }
};

class ParticleLSDM
{
public:
  ParticleLSDM() = default;
  ~ParticleLSDM() = default;

  // The velocit for a particle for a given iteration.
  vec3 velMean{};

  // The velocity fluctuation for a particle for a given iteration.
  // Starts out as the initial value until a particle is "released" into the domain
  vec3 velFluct{};
  // difference between the current and last iteration of the uFluct variable
  vec3 delta_velFluct{};

  // The velocity fluctuation for a particle from the last iteration
  vec3 velFluct_old{};

  // stress tensor from the last iteration (6 component because stress tensor is symmetric)
  mat3sym tau{};
  //
  float CoEps{};
  float nuT{};

  void reset(const vec3 &velFluct0, const mat3sym &tau0)
  {
    // note: velMean, CoEps, and nuT are set by the interpolation, no need to reset them.
    velFluct = velFluct0;
    velFluct_old = velFluct0;
    delta_velFluct = { 0.0, 0.0, 0.0 };

    tau = tau0;
  }
};

class ParticleMetadata
{
public:
  ParticleMetadata() = default;
  ~ParticleMetadata() = default;

  // particle type
  ParticleType type{};
  // the initial position for the particle
  vec3 pos_init{};
  // the time of release for the particle (need to change to QEStime)
  QEStime time_start{};
  // the index of the source the particle came from
  int source_id{};
  // initial particle mass [g]
  float m_init{};

  void reset(const ParticleType &t, const vec3 &p0, const QEStime &t0, const int &sID, const float &m0)
  {
    type = t;
    pos_init = p0;
    time_start = t0;
    source_id = sID;
    m_init = m0;
  }
};


class Particle
{
private:
  Particle()
    : type(ParticleType::tracer), state(INACTIVE),
      d(0.0), m(0.0), m_o(0.0), rho(0.0),
      c1(2.049), c2(1.19), depFlag(false), decayConst(0.0), wdecay(1.0)
  {
  }


public:
  // initializer
  Particle(const bool &flag, const ParticleType &type_in)
    : type(type_in), state(INACTIVE),
      d(0.0), m(0.0), m_o(0.0), rho(0.0),
      c1(2.049), c2(1.19), depFlag(flag), decayConst(0.0), wdecay(1.0),
      sourceIdx(-1)
  {}
  explicit Particle(const ParticleType &type_in)
    : type(type_in), state(INACTIVE),
      d(0.0), m(0.0), m_o(0.0), rho(0.0),
      c1(2.049), c2(1.19), depFlag(false), decayConst(0.0), wdecay(1.0),
      sourceIdx(-1)
  {}
  /*
    // initializer
    Particle(const bool &flag, const ParticleType &type, const float &d_p, const float &m_p, const float &rho_p)
      : particleType(type),
        d(d_p), m(m_p), m_o(m_p), rho(rho_p),
        c1(2.049), c2(1.19), depFlag(flag), decayConst(0.0), wdecay(1.0)
    {
    }
  */

  // particle type
  ParticleType type;
  // state of the particle
  ParticleState state;

public:
  // destructor
  virtual ~Particle() = default;

  // ParticleType particleType;// particle type
  //  std::string tag;// particle type tag

  // the initial position for the particle, to not be changed after the simulation starts
  vec3 pos_init;

  // the time of release for the particle
  float tStrt{};
  QEStime timeStrt{};
  // id of particle (for tracking purposes)
  uint32_t ID{};

  // the index of the source the particle came from
  int sourceIdx;


  // once initial positions are known, can set these values using urb and turb info
  // Initially, the initial x component of position for the particle.
  // After the solver starts to run, the current x component of position for the particle.
  vec3 pos;

  // The velocit for a particle for a given iteration.
  vec3 velMean;

  // The velocity fluctuation for a particle for a given iteration.
  // Starts out as the initial value until a particle is "released" into the domain
  vec3 velFluct;

  // Particle displacements for each time step (not used)
  // vec3 dist;

  // Total velocities (mean and fluctuation) for each time step (not used)

  // vec3 velTot;

  float CoEps{};
  float nuT{};

  // The velocity fluctuation for a particle from the last iteration
  vec3 velFluct_old;

  // stress tensor from the last iteration (6 component because stress tensor is symmetric)
  mat3sym tau;

  // difference between the current and last iteration of the uFluct variable
  vec3 delta_velFluct;

  // bool isRogue = false;// this is false until it becomes true. Should not go true.
  // bool isActive = false;// this is true until it becomes false.

  // particle physical property
  float d;// particle diameter diameter [microns]
  // float d_m;// particle diameter diameter [m]
  float m;// particle mass [g]
  // float m_kg;// particle mass [kg]
  float m_o;// initial particle mass [g]
  // float m_kg_o;// initial particle mass [kg]
  float rho;// density of particle

  // deposition variables
  float c1;// Stk* fit param (exponent)
  float c2;// Stk* fit param (exponent)
  // float Sc{};// Schmidt number
  // float taud{};// characteristic relaxation time [s]
  // float vd{};// deposition velocity [m/s]
  bool depFlag;// whether a particle deposits

  // deposition container
  bool dep_buffer_flag{};
  float decayConst;// mass decay constant

  // settling variables (only use as local variables)
  // float dstar = 0;// dimensionless grain diameter
  // float Cd = 0;// drag coefficient
  // double wstar = 0;// dimensionless settling velocity
  // double vs = 0;// settling velocity [m/s]

  // decay variables
  double wdecay;// (1 - fraction) particle decayed [0,1]
};
