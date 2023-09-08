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

/** @file SourceType.hpp
 * @brief  This class represents a generic sourece type
 *
 * @note Pure virtual child of ParseInterface
 * @sa ParseInterface
 */

#include "Source.hpp"

void ParseSource::setReleaseType()
{
  // this variable is a temporary variable to set the publicly available variable m_rType.
  // !!! To make this happen, each source is expected to call the function setReleaseType() inside their call of the function parseValues()
  //  setReleaseType uses parseMultiPolymorph() to fill this variable, then checks to make sure it is size 1 as only 1 release type is allowed,
  //  then setReleaseType() sets the variable m_rType to be the one value found in this variable.
  std::vector<ReleaseType *> rType_tmp;

  // first parse all the release types into the temporary variable rType_tmp
  parseMultiPolymorphs(false, rType_tmp, Polymorph<ReleaseType, ReleaseType_instantaneous>("ReleaseType_instantaneous"));
  parseMultiPolymorphs(false, rType_tmp, Polymorph<ReleaseType, ReleaseType_continuous>("ReleaseType_continuous"));
  parseMultiPolymorphs(false, rType_tmp, Polymorph<ReleaseType, ReleaseType_duration>("ReleaseType_duration"));

  // now if the number of release types is not 1, there was a problem, need to quit with an error
  if (rType_tmp.empty()) {
    std::cerr << "ERROR (SourceType::setReleaseType): there was no input releaseType!" << std::endl;
    exit(1);
  }
  if (rType_tmp.size() > 1) {
    std::cerr << "ERROR (SourceType::setReleaseType): there was more than one input releaseType!" << std::endl;
    exit(1);
  }

  // the number of release types is 1, so now set the public release type to be the one that we have
  m_rType = rType_tmp.at(0);
}

// This function uses the temporary variable rType_tmp to parse all the release types found in the .xml file for a given source,
// then checks to make sure rType_tmp is size 1 as only 1 release type is allowed,
// finally this function sets the public variable m_rType to be the one value found in rType_tmp.
// !!! To make this happen, each source is expected to call the function setReleaseType() inside their call of the function parseValues().
// LA-notes: it may be possible to move rType_tmp into this function,
//  but I'm not sure how two pointers pointing to the same variable will act once out of scope.
void ParseSource::setSourceGeometry()
{
  // this variable is a temporary variable to set the publicly available variable m_rType.
  // !!! To make this happen, each source is expected to call the function setReleaseType() inside their call of the function parseValues()
  //  setReleaseType uses parseMultiPolymorph() to fill this variable, then checks to make sure it is size 1 as only 1 release type is allowed,
  //  then setReleaseType() sets the variable m_rType to be the one value found in this variable.
  std::vector<SourceGeometry *> sGeom_tmp;

  // first parse all the release types into the temporary variable rType_tmp
  parseMultiPolymorphs(false, sGeom_tmp, Polymorph<SourceGeometry, SourceGeometry_Cube>("SourceCube"));
  parseMultiPolymorphs(false, sGeom_tmp, Polymorph<SourceGeometry, SourceGeometry_FullDomain>("SourceFullDomain"));
  parseMultiPolymorphs(false, sGeom_tmp, Polymorph<SourceGeometry, SourceGeometry_Line>("SourceLine"));
  parseMultiPolymorphs(false, sGeom_tmp, Polymorph<SourceGeometry, SourceGeometry_Point>("SourcePoint"));
  parseMultiPolymorphs(false, sGeom_tmp, Polymorph<SourceGeometry, SourceGeometry_SphereShell>("SourceSphereSurface"));


  // now if the number of release types is not 1, there was a problem, need to quit with an error
  if (sGeom_tmp.empty()) {
    std::cerr << "ERROR (SourceType::setReleaseType): there was no input releaseType!" << std::endl;
    exit(1);
  }
  if (sGeom_tmp.size() > 1) {
    std::cerr << "ERROR (SourceType::setReleaseType): there was more than one input releaseType!" << std::endl;
    exit(1);
  }

  // the number of release types is 1, so now set the public release type to be the one that we have
  m_sourceType = sGeom_tmp.at(0);
}

void ParseSource::setParticleType()
{
  std::vector<ParseParticle *> protoParticle_tmp;

  parseMultiPolymorphs(false, protoParticle_tmp, Polymorph<ParseParticle, ParseParticleTracer>("ParticleTracer"));
  parseMultiPolymorphs(false, protoParticle_tmp, Polymorph<ParseParticle, ParseParticleSmall>("ParticleSmall"));
  parseMultiPolymorphs(false, protoParticle_tmp, Polymorph<ParseParticle, ParseParticleLarge>("ParticleLarge"));
  parseMultiPolymorphs(false, protoParticle_tmp, Polymorph<ParseParticle, ParseParticleHeavyGas>("ParticleHeavyGas"));

  if (protoParticle_tmp.empty()) {
    // std::cerr << "ERROR (SourceType::setParticleType): there was no input particle type!" << std::endl;
    // exit(1);
    m_protoParticle = new ParseParticleTracer();
    return;
  } else if (protoParticle_tmp.size() > 1) {
    std::cerr << "ERROR (SourceType::setParticleType): there was more than one input particle type!" << std::endl;
    exit(1);
  }

  // the number of release types is 1, so now set the public release type to be the one that we have
  m_protoParticle = protoParticle_tmp.at(0);
}