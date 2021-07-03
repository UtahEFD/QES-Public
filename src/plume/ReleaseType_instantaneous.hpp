
#pragma once


#include "ReleaseType.hpp"


class ReleaseType_instantaneous : public ReleaseType
{
private:
  // note that this also inherits data members ParticleReleaseType m_rType, int m_parPerTimestep, double m_releaseStartTime,
  //  double m_releaseEndTime, and int m_numPar from ReleaseType.
  // guidelines for how to set these variables within an inherited ReleaseType are given in ReleaseType.hpp.

  int numPar;


protected:
public:
  // Default constructor
  ReleaseType_instantaneous()
  {
  }

  // destructor
  ~ReleaseType_instantaneous()
  {
  }


  virtual void parseValues()
  {
    parReleaseType = ParticleReleaseType::instantaneous;

    parsePrimitive<int>(true, numPar, "numPar");
  }


  void calcReleaseInfo(const double &timestep, const double &simDur)
  {
    // set the overall releaseType variables from the variables found in this class
    m_parPerTimestep = numPar;
    m_releaseStartTime = 0;
    m_releaseEndTime = 0;
    m_numPar = numPar;
  }
};
