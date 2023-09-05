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

/** @file ParticleFactories.cpp
 * @brief
 */

#include "ParticleFactories.hpp"

ParticleTypeFactory::ParticleTypeFactory()
{
  std::string tracerstr = "ParticleTracer";
  std::string smallstr = "ParticleSmall";
  std::string largestr = "ParticleLarge";
  std::string heavygasstr = "ParticleHeavyGas";

  RegisterParticles(tracerstr, &particleTracerFactory);
  RegisterParticles(smallstr, &particleSmallFactory);
  RegisterParticles(largestr, &particleLargeFactory);
  RegisterParticles(heavygasstr, &particleHeavyGasFactory);
}

void ParticleTypeFactory::RegisterParticles(const std::string &particleType, ParticleFactory *particleFactory)
{
  // std::cout <<
  ParticleTypeContainer.insert(std::pair<std::string, ParticleFactory *>(particleType, particleFactory));
}

// Function to return the actual particle object
Particle *ParticleTypeFactory::Create(const ParseParticle *proptoParticle)
{
  /*
  std::cout << "Element of ParticleTypeContainer are: " << std::endl;
  for (auto const &pair : ParticleTypeContainer) {
    std::cout << "{" << pair.first << ": " << pair.second << "}\n";
  }
  std::cout << " ParticleTypeContainer.at(proptoParticle->tag) is: "
            << ParticleTypeContainer.at(proptoParticle->tag) << std::endl;
  */
  //      std::cout << "Calling create() from the " << proptoParticle->tag << " factory" << std::endl;
  return ParticleTypeContainer.at(proptoParticle->tag)->create();
  //      std::cout << "done calling create() from the " << proptoParticle->tag << " factory" << std::endl;
}