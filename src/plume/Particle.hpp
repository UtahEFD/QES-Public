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

#include "util/ParseInterface.h"

enum ParticleType {
  tracer,
  heavy,
  large,
  heavygas
};

class Particle;

class ParseParticle : public ParseInterface
{
protected:
public:
  // particle type
  ParticleType particleType;
  std::string tag;

  // Physical properties
  double d, d_m, m, m_kg, rho;
  bool depFlag;
  double decayConst, c1, c2;

  ParseParticle(const bool &flag, std::string str, const ParticleType &type)
    : d(0.0), d_m(0.0), m(0.0), m_kg(0.0), rho(0.0),
      depFlag(flag), decayConst(0.0), c1(2.049), c2(1.19),
      tag(std::move(str)), particleType(type)
  {}

  // destructor
  ~ParseParticle() = default;

  virtual void parseValues() = 0;

  virtual void setParticleParameters(Particle *) = 0;

private:
  // default constructor
  ParseParticle()
    : d(0.0), d_m(0.0), m(0.0), m_kg(0.0), rho(0.0),
      depFlag(false), decayConst(0.0), c1(2.049), c2(1.19),
      particleType(ParticleType::tracer), tag("ParticleTracer")
  {}
};

class Particle
{
private:
  Particle()
    : d(0.0), d_m(0.0), m(0.0), m_kg(0.0), m_o(0.0), m_kg_o(0.0), rho(0.0),
      decayConst(0.0), c1(2.049), c2(1.19), wdecay(1.0),
      depFlag(false), particleType(ParticleType::tracer), tag("ParticleTracer")
  {
  }

protected:
public:
  // initializer
  Particle(const bool &flag, std::string str, const ParticleType &type)
    : d(0.0), d_m(0.0), m(0.0), m_kg(0.0), m_o(0.0), m_kg_o(0.0), rho(0.0),
      decayConst(0.0), c1(2.049), c2(1.19), wdecay(1.0),
      tag(std::move(str)), depFlag(flag), particleType(type)
  {
  }

  // initializer
  Particle(const double &d_part, const double &m_part, const double &rho_part)
  {
    // diameter of particle (micron and m)
    d = d_part;
    d_m = (1.0E-6) * d;

    // mass of particle (g and kg)
    m = m_part;
    m_kg = (1.0E-3) * m;

    // density of particle
    rho = rho_part;

    // tag
    tag = "ParticleTracer";// tagged as "tracer" so when particle type is unspecified in XML, it defaults to tracer

    // (1 - fraction) particle deposited
    depFlag = false;
    decayConst = 0;
    c1 = 2.049;
    c2 = 1.19;
    // (1 - fraction) particle deposited
    wdecay = 1.0;
  }

  // destructor
  virtual ~Particle() = default;

  ParticleType particleType;// particle type
  std::string tag;// particle type tag

  // the initial position for the particle, to not be changed after the simulation starts
  double xPos_init{};// the initial x component of position for the particle
  double yPos_init{};// the initial y component of position for the particle
  double zPos_init{};// the initial z component of position for the particle

  double tStrt{};// the time of release for the particle
  int particleID{};// id of particle (for tracking purposes)
  int sourceIdx{};// the index of the source the particle came from

  // once initial positions are known, can set these values using urb and turb info
  // Initially, the initial x component of position for the particle.
  // After the solver starts to run, the current x component of position for the particle.
  double xPos{};// x component of position for the particle.
  double yPos{};// y component of position for the particle.
  double zPos{};// z component of position for the particle.

  // The velocit for a particle for a given iteration.
  double uMean{};// u component
  double vMean{};// v component
  double wMean{};// w component

  // The velocity fluctuation for a particle for a given iteration.
  // Starts out as the initial value until a particle is "released" into the domain
  double uFluct{};// u component
  double vFluct{};// v component
  double wFluct{};// w component

  // Particle displacements for each time step
  double disX{};
  double disY{};
  double disZ{};

  // Total velocities (mean and fluctuation) for each time step
  double uTot{};
  double vTot{};
  double wTot{};

  double CoEps{};

  // The velocity fluctuation for a particle from the last iteration
  double uFluct_old{};// u component
  double vFluct_old{};// v component
  double wFluct_old{};// w component

  // stress tensor from the last iteration (6 component because stress tensor is symmetric)
  double txx_old{};// this is the stress in the x direction on the x face from the last iteration
  double txy_old{};// this is the stress in the y direction on the x face from the last iteration
  double txz_old{};// this is the stress in the z direction on the x face from the last iteration
  double tyy_old{};// this is the stress in the y direction on the y face from the last iteration
  double tyz_old{};// this is the stress in the z direction on the y face from the last iteration
  double tzz_old{};// this is the stress in the z direction on the z face from the last iteration

  double delta_uFluct{};// this is the difference between the current and last iteration of the uFluct variable
  double delta_vFluct{};// this is the difference between the current and last iteration of the vFluct variable
  double delta_wFluct{};// this is the difference between the current and last iteration of the wFluct variable

  bool isRogue{};// this is false until it becomes true. Should not go true.
  bool isActive = false;// this is true until it becomes false.

  // particle physical property
  double d;// particle diameter diameter [microns]
  double d_m;// particle diameter diameter [m]
  double m;// particle mass [g]
  double m_kg;// particle mass [kg]
  double m_o;// initial particle mass [g]
  double m_kg_o;// initial particle mass [kg]
  double rho;// density of particle

  // deposition variables
  double Sc{};// Schmidt number
  double taud{};// characteristic relaxation time [s]
  double vd{};// deposition velocity [m/s]
  bool depFlag;// whether a particle deposits

  // deposition container
  bool dep_buffer_flag{};
  std::vector<int> dep_buffer_cell;
  std::vector<float> dep_buffer_val;

  double decayConst;// mass decay constant
  double c1;// Stk* fit param (exponent)
  double c2;// Stk* fit param (exponent)
  // settling variables
  double dstar = 0;// dimensionless grain diameter
  double Cd = 0;// drag coefficient
  double wstar = 0;// dimensionless settling velocity
  double vs = 0;// settling velocity [m/s]

  // decay variables
  double wdecay;// (1 - fraction) particle decayed [0,1]

  virtual void setSettlingVelocity(const double &, const double &){};
};
