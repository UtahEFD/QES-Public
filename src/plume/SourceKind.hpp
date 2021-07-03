//
//  SourceKind.hpp
//
//  This class represents a generic source
//
//  Created by Jeremy Gibbs on 03/28/19.
//  Updated by Loren Atwood on 01/31/20.
//

#pragma once

#include <random>
#include <list>

#include "Particle.hpp"
#include "ReleaseType.hpp"
#include "ReleaseType_instantaneous.hpp"
#include "ReleaseType_continuous.hpp"
#include "ReleaseType_duration.hpp"

#include "util/ParseInterface.h"


enum SourceShape {
  point,
  line,
  circle,
  cube,
  fullDomain
};

class SourceKind : public ParseInterface
{
protected:
  // this variable is a temporary variable to be used by setReleaseType() to set the publicly available variable m_rType.
  // !!! To make this happen, each source is expected to call the function setReleaseType() inside their call of the function parseValues()
  //  setReleaseType uses parseMultiPolymorph() to fill this variable, then checks to make sure it is size 1 as only 1 release type is allowed,
  //  then setReleaseType() sets the variable m_rType to be the one value found in this variable.
  std::vector<ReleaseType *> rType_tmp;


public:
  // this is the index of the source in the dispersion class overall list of sources
  // this is used to set the source ID for a given particle, to know from which source each particle comes from
  // !!! this will only be set correctly if a call to setSourceIdx() is done by the class that sets up a vector of this class.
  int sourceIdx;

  // this is a description variable for determining the source shape. May or may not be used.
  // !!! this needs set by parseValues() in each source generated from input files.
  SourceShape m_sShape;

  // this is a pointer to the release type, which is expected to be chosen by parseValues() by each source via a call to setReleaseType().
  // this data structure holds information like the total number of particles to be released by the source, the number of particles to release
  // per time for each source, and the start and end times to be releasing from the source.
  // !!! this needs set by parseValues() in each source generated from input files by a call to the setReleaseType() function
  ReleaseType *m_rType;

  // LA-future work: need a class similar to ReleaseType that describes the input source mass.
  //  This could be mass, mass per time, volume with a density, and volume per time with a density.

  // LA-future work: need a class similar to ReleaseType that describes how to distribute the particles along the source geometry
  //  This could be uniform, random normal distribution, ... I still need to think of more distributions.
  // On second thought, this may not be possible to do like ReleaseType, it may need to be specific to each source geometry.
  //  so maybe it needs to be more like BoundaryConditions, where each source determines which pointer function to choose for emitting particles
  //  based on a string input describing the distribution. Really depends on how easy the implementation details become.


  // default constructor
  SourceKind()
  {
  }

  // destructor
  virtual ~SourceKind()
  {
  }


  // This function uses the temporary variable rType_tmp to parse all the release types found in the .xml file for a given source,
  // then checks to make sure rType_tmp is size 1 as only 1 release type is allowed,
  // finally this function sets the public variable m_rType to be the one value found in rType_tmp.
  // !!! To make this happen, each source is expected to call the function setReleaseType() inside their call of the function parseValues().
  // LA-notes: it may be possible to move rType_tmp into this function,
  //  but I'm not sure how two pointers pointing to the same variable will act once out of scope.
  void setReleaseType()
  {
    // first parse all the release types into the temporary variable rType_tmp
    parseMultiPolymorphs(false, rType_tmp, Polymorph<ReleaseType, ReleaseType_instantaneous>("ReleaseType_instantaneous"));
    parseMultiPolymorphs(false, rType_tmp, Polymorph<ReleaseType, ReleaseType_continuous>("ReleaseType_continuous"));
    parseMultiPolymorphs(false, rType_tmp, Polymorph<ReleaseType, ReleaseType_duration>("ReleaseType_duration"));

    // now if the number of release types is not 1, there was a problem, need to quit with an error
    if (rType_tmp.size() == 0) {
      std::cerr << "ERROR (SourceKind::setReleaseType): there was no input releaseType!" << std::endl;
      exit(1);
    }
    if (rType_tmp.size() > 1) {
      std::cerr << "ERROR (SourceKind::setReleaseType): there was more than one input releaseType!" << std::endl;
      exit(1);
    }

    // the number of release types is 1, so now set the public release type to be the one that we have
    m_rType = rType_tmp.at(0);
  }


  // this function is used to parse all the variables for each source from the input .xml file
  // each source overloads this function with their own version, allowing different combinations of input variables for each source,
  // all these differences handled by parseInterface().
  // The = 0 at the end should force each inheriting class to require their own version of this function
  // !!! in order for all the different combinations of input variables to work properly for each source, this function requires calls to the
  //  setReleaseType() function and manually setting the variable m_sShape in each version found in sources that inherit from this class.
  //  This is in addition to any other variables required for an individual source that inherits from this class.
  virtual void parseValues() = 0;


  // this function is used for setting the sourceIdx variable of this class, and has to be called by the class that sets up a vector of this class.
  // this is required since each source does NOT know which index it has in the list without information from outside the parsing classes
  // !!! this needs called by the class that sets up a vector of this class, using the index of the source from the vector
  // LA note: could set this value directly without a function call, but this makes it seem more deliberate
  void setSourceIdx(const int &sourceIdx_val)
  {
    sourceIdx = sourceIdx_val;
  }


  // this function is for checking the source metadata to make sure all particles will be released within the domain.
  // There is one source so far (SourceFullDomain) that actually uses this function to set a few metaData variables
  //  specific to that source as well as to do checks to make sure particles stay within the domain. This is not a problem
  //  so long as it is done this way with with future sources very sparingly.
  //  In other words, avoid using this function to set variables unless you have to.
  // !!! each source needs to have this function manually called for them by whatever class sets up a vector of this class.
  virtual void checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend) = 0;


  // this function is for appending a new set of particles to the provided vector of particles
  // the way this is done differs for each source inheriting from this class, but in general
  //  the idea is to determine whether the input time is within the time range particles should be released
  //  then the particle positions are set using the particles to release per time, geometry information, and
  //  distribution information.
  // !!! only the particle initial positions, release time, and soureIdx are set for each particle,
  //  other particle information needs set by whatever called this function
  //  right after the call to this function to make it work correctly.
  // Note that this is a pure virtual function - enforces that the derived class MUST define this function
  //  this is done by the = 0 at the end of the function
  // LA-future work: There is still room for improvement to this function for each different source.
  //  It requires determining additional input .xml file information for each source, which still needs worked out.
  // LA-other notes: currently this is outputting the number of particles to release per time, which is the number of particles
  //  appended to the list. According to Pete, the int output could be used for returning error messages,
  //  kind of like the exit success or exit failure return methods.
  // !!! Because the input vector is never empty if there is more than one source,
  //   the size of the vector should NOT be used for output for this function!
  //  In order to make this function work correctly, the number of particles to release per timestep needs to be the output
  virtual int emitParticles(const float dt, const float currTime, std::list<Particle *> &emittedParticles) = 0;
};
