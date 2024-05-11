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

/** @file ParticleFactories.hpp
 * @brief
 */

#pragma once

#include "Particle.hpp"
#include "ParticleTracer.hpp"
#include "ParticleSmall.hpp"
#include "ParticleLarge.hpp"
#include "ParticleHeavyGas.hpp"
#include <unordered_map>

class ParticleFactory
{

public:
  virtual ~ParticleFactory() {}
  virtual Particle *create() = 0;
};


class ParticleTracerFactory : public ParticleFactory
{
public:
  Particle *create() override
  {
    //    std::cout << "Creating new ParticleTracer" << std::endl;
    return new ParticleTracer();
    //    std::cout << "Done creating new ParticleTracer" << std::endl;
  }
};

class ParticleSmallFactory : public ParticleFactory
{

public:
  Particle *create() override
  {
    return new ParticleSmall();
  }
};

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

class ParticleTypeFactory
{

private:
  std::unordered_map<std::string, ParticleFactory *> ParticleTypeContainer;

  ParticleTracerFactory particleTracerFactory;
  ParticleSmallFactory particleSmallFactory;
  ParticleLargeFactory particleLargeFactory;
  ParticleHeavyGasFactory particleHeavyGasFactory;

public:
  ParticleTypeFactory();

  ~ParticleTypeFactory()
  {
  }

  // Function to read in all possible particle types and create factories for them
  void RegisterParticles(const std::string &particleType, ParticleFactory *particleFactory);

  // Function to return the actual particle object
  // Particle *Create(const std::string &particleType);
  Particle *Create(const ParseParticle *proptoParticle);
};
