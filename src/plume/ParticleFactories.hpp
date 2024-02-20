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

/** @file ParticleFactories.hpp
 * @brief
 */

#pragma once

#include "Particle.h"
#include "Particle_Tracer.hpp"
#include "Particle_Heavy.hpp"
#include <unordered_map>

class ParticleFactory
{

public:
  virtual ~ParticleFactory() {}
  virtual Particle *create() = 0;
};


class Particle_TracerFactory : public ParticleFactory
{
public:
  Particle *create() override
  {
    //    std::cout << "Creating new Particle_Tracer" << std::endl;
    return new Particle_Tracer();
    //    std::cout << "Done creating new Particle_Tracer" << std::endl;
  }
};

class Particle_HeavyFactory : public ParticleFactory
{

public:
  Particle *create() override
  {
    return new Particle_Heavy();
  }
};

/*
class ParticleLargeFactory : public ParticleFactory
{

public:
  Particle *create() override
  {
    return new ParticleLarge();
  }
};

class ParticleHeavyGasFactory : public ParticleFactory
{

public:
  Particle *create() override
  {
    return new ParticleHeavyGas();
  }
};
*/
class ParticleTypeFactory
{

private:
  std::unordered_map<ParticleType, ParticleFactory *, std::hash<int>> ParticleTypeContainer;

  Particle_TracerFactory particle_TracerFactory;
  Particle_HeavyFactory particle_HeavyFactory;
  // ParticleLargeFactory particleLargeFactory;
  // ParticleHeavyGasFactory particleHeavyGasFactory;

public:
  ParticleTypeFactory();

  ~ParticleTypeFactory()
  {
  }

  // Function to read in all possible particle types and create factories for them
  void RegisterParticles(const ParticleType &, ParticleFactory *);

  // Function to return the actual particle object
  // Particle *Create(const std::string &particleType);
  Particle *Create(const ParseParticle *proptoParticle);
};
