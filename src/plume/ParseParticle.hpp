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

/** @file ParseParticle.hpp 
 * @brief 
 *  
 *
 * @note Pure virtual child of ParseInterface 
 * @sa 
 */

#pragma once

#include <cmath>

#include "util/ParseInterface.h"
//#include "Particle.hpp"
//#include "ParticleTracer.hpp"
//#include "ParticleSmall.hpp"
//#include "ParticleLarge.hpp"
//#include "ParticleHeavyGas.hpp"


enum ParticleType {
  tracer,
  small,
  large,
  heavygas
};


class ParseParticle : public ParseInterface
{
protected:
public:
  ParticleType parType;

  //  std::vector<Particle *> protoParticle_tmp;
  //  Particle *protoParticle; // prototype particle with values from XML, to be copied to emitted particles


  // Physical properties
  double d;
  double d_m;
  double m;
  double m_kg;
  double rho;
  bool depFlag;
  std::string tag;
  /*
  // the initial position for the particle, to not be changed after the simulation starts
  double xPos_init;// the initial x component of position for the particle
  double yPos_init;// the initial y component of position for the particle
  double zPos_init;// the initial z component of position for the particle

  double tStrt;// the time of release for the particle
  int particleID;// id of particl (for tracking purposes)
  int sourceIdx;// the index of the source the particle came from

  // once initial positions are known, can set these values using urb and turb info
  // Initially, the initial x component of position for the particle.
  // After the solver starts to run, the current x component of position for the particle.
  double xPos;// x component of position for the particle.
  double yPos;// y component of position for the particle.
  double zPos;// z component of position for the particle.

  // The velocit for a particle for a given iteration.
  double uMean;// u component
  double vMean;// v component
  double wMean;// w component

  // The velocity fluctuation for a particle for a given iteration.
  // Starts out as the initial value until a particle is "released" into the domain
  double uFluct;// u component
  double vFluct;// v component
  double wFluct;// w component

  // Particle displacements for each time step
  double disX;
  double disY;
  double disZ;

  // Total velocities (mean plus fluct) for each time step
  double uTot;
  double vTot;
  double wTot;

  double CoEps;

  // The velocity fluctuation for a particle from the last iteration
  double uFluct_old;// u component
  double vFluct_old;// v component
  double wFluct_old;// w component

  // stress tensor from the last iteration (6 component because stress tensor is symetric)
  double txx_old;// this is the stress in the x direction on the x face from the last iteration
  double txy_old;// this is the stress in the y direction on the x face from the last iteration
  double txz_old;// this is the stress in the z direction on the x face from the last iteration
  double tyy_old;// this is the stress in the y direction on the y face from the last iteration
  double tyz_old;// this is the stress in the z direction on the y face from the last iteration
  double tzz_old;// this is the stress in the z direction on the z face from the last iteration

  double delta_uFluct;// this is the difference between the current and last iteration of the uFluct variable
  double delta_vFluct;// this is the difference between the current and last iteration of the vFluct variable
  double delta_wFluct;// this is the difference between the current and last iteration of the wFluct variable

  bool isRogue;// this is false until it becomes true. Should not go true. It is whether a particle has gone rogue or not
  bool isActive;// this is true until it becomes false.  If a particle leaves the domain or runs out of mass, this becomes false.

  // deposition vatiables
  double wdepos;// (1 - fraction) particle deposited [0,1]
  double Sc;// Schmidt number
  double taud;// characteristic relaxation time [s]
  double vd;// deposition velocity [m/s]
  bool depFlag; // whether a particle deposits

  // settling vatiables
  double dstar;// dimensionless grain diameter
  double Cd;// drag coefficent
  double wstar;// dimensionless settling velocity
  double vs;// settling velocity [m/s]

  // decay varables
  double wdecay;// (1 - fraction) particle decayed [0,1]
*/

  // default constructor
  ParseParticle()
  {
    d = 0.0;
    d_m = 0.0;
    m = 0.0;
    m_kg = 0.0;
    rho = 0.0;
    depFlag = false;
    tag = "ParticleTracer";
    d_m = d * (1.0E-6);
    m_kg = m * (1.0E-3);
  }

  // destructor
  ~ParseParticle()
  {
  }

  virtual void parseValues()
  {
    parType = ParticleType::tracer;
  };

  //  virtual void setSettlingVelocity(const double &, const double &) {};
};


class ParseParticleTracer : public ParseParticle
{
protected:
public:
  ParticleType parType;

  //  std::vector<ParseParticle *> protoParticle_tmp;
  //  ParseParticle *protoParticle; // prototype particle with values from XML, to be copied to emitted particles

  /*
  // Physical properties
  double d;
  double d_m;
  double m;
  double m_kg;
  double rho;
  bool depFlag;
*/
  // default constructor
  ParseParticleTracer()
  {
    d = 0.0;
    d_m = 0.0;
    m = 0.0;
    m_kg = 0.0;
    rho = 0.0;
    depFlag = false;
    tag = "ParticleTracer";
  }

  // destructor
  ~ParseParticleTracer()
  {
  }

  virtual void parseValues()
  {
    parType = ParticleType::tracer;
  }
};


class ParseParticleSmall : public ParseParticle
{
protected:
public:
  ParticleType parType;

  //  std::vector<ParseParticle *> protoParticle_tmp;
  //  ParseParticle *protoParticle; // prototype particle with values from XML, to be copied to emitted particles

  /*
  // Physical properties
  double d;
  double d_m;
  double m;
  double m_kg;
  double rho;
  bool depFlag;
*/
  // default constructor
  ParseParticleSmall()
  {
    d = 0.0;
    d_m = 0.0;
    m = 0.0;
    m_kg = 0.0;
    rho = 0.0;
    depFlag = true;
    tag = "ParticleSmall";
  }

  // destructor
  ~ParseParticleSmall()
  {
  }

  virtual void parseValues()
  {
    parType = ParticleType::small;
    parsePrimitive<double>(true, rho, "particleDensity");
    parsePrimitive<double>(true, d, "particleDiameter");
    parsePrimitive<bool>(false, depFlag, "depositionFlag");
    d_m = d * (1.0E-6);
    m_kg = m * (1.0E-3);
  }
};


class ParseParticleLarge : public ParseParticle
{
protected:
public:
  ParticleType parType;

  //  std::vector<ParseParticle *> protoParticle_tmp;
  //  ParseParticle *protoParticle; // prototype particle with values from XML, to be copied to emitted particles

  /*
  // Physical properties
  double d;
  double d_m;
  double m;
  double m_kg;
  double rho;
  bool depFlag;
*/
  // default constructor
  ParseParticleLarge()
  {
    d = 0.0;
    d_m = 0.0;
    m = 0.0;
    m_kg = 0.0;
    rho = 0.0;
    depFlag = true;
    tag = "ParticleLarge";
  }

  // destructor
  ~ParseParticleLarge()
  {
  }

  virtual void parseValues()
  {
    parType = ParticleType::large;
    parsePrimitive<double>(true, rho, "particleDensity");
    parsePrimitive<double>(true, d, "particleDiameter");
    parsePrimitive<bool>(false, depFlag, "depositionFlag");
    d_m = d * (1.0E-6);
    m_kg = m * (1.0E-3);
  }
};


class ParseParticleHeavyGas : public ParseParticle
{
protected:
public:
  ParticleType parType;

  //  std::vector<ParseParticle *> protoParticle_tmp;
  //  ParseParticle *protoParticle; // prototype particle with values from XML, to be copied to emitted particles

  /*
  // Physical properties
  double d;
  double d_m;
  double m;
  double m_kg;
  double rho;
  bool depFlag;
*/
  // default constructor
  ParseParticleHeavyGas()
  {
    d = 0.0;
    d_m = 0.0;
    m = 0.0;
    m_kg = 0.0;
    rho = 0.0;
    depFlag = true;
    tag = "ParticleHeavyGas";
  }

  // destructor
  ~ParseParticleHeavyGas()
  {
  }

  virtual void parseValues()
  {
    parType = ParticleType::heavygas;
    parsePrimitive<double>(true, rho, "particleDensity");
    parsePrimitive<double>(true, d, "particleDiameter");
    parsePrimitive<bool>(false, depFlag, "depositionFlag");
    d_m = d * (1.0E-6);
    m_kg = m * (1.0E-3);
  }
};
