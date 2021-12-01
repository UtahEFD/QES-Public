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

/** @file ParticleFactories.hpp 
 * @brief 
 */

// #pragma once

#include "Particle.hpp"
#include <unordered_map>

class ParticleFactory
{

public:
  virtual ~ParticleFactory() {}
  virtual Particle* create() = 0;
};


class ParticleTracerFactory : public ParticleFactory
{

public:

  virtual Particle* create()
  {
    std::cout << "Creating new ParticleTracer" << std::endl;
    return new ParticleTracer();
    std::cout << "Done creating new ParticleTracer" << std::endl;
  }
};

class ParticleSmallFactory : public ParticleFactory
{

public:
  virtual Particle* create()
  {
    return new ParticleSmall();       
  }

};

class ParticleLargeFactory : public ParticleFactory
{

public:
  virtual Particle* create()
  {
    return new ParticleLarge();       
  }

};

class ParticleHeavyGasFactory : public ParticleFactory
{

public:
  virtual Particle* create()
  {
    return new ParticleHeavyGas();       
  }

};

class ParticleTypeFactory
{
 
  private:
    std::unordered_map<std::string, ParticleFactory *> ParticleTypeContainer;
  
  public:
    ParticleTypeFactory()
    {
    }

    ~ParticleTypeFactory()
    {
    }
    
    // Function to read in all possible particle types and create factories for them
    void RegisterParticles(std::string particleType, ParticleFactory * particleFactory)
    {
      //std::cout << 
      ParticleTypeContainer.insert(std::pair<std::string, ParticleFactory*>(particleType, particleFactory));
      
    }
    
    // Function to return the actual particle object 
    Particle * Create(std::string particleType)
    {
      std::cout << "Element of ParticleTypeContainer are: " << std::endl;
      for (auto const &pair: ParticleTypeContainer) {
        std::cout << "{" << pair.first << ": " << pair.second << "}\n";
      }
      std::cout << " ParticleTypeContainer.at(particleType) is: " << ParticleTypeContainer.at(particleType) << std::endl;
      
      std::cout << "Calling create() from the " << particleType << " factory" << std::endl;
      return ParticleTypeContainer.at(particleType)->create();
      std::cout << "done calling create() from the " << particleType << " factory" << std::endl;
    }


};



