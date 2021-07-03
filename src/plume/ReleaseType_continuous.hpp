
#pragma once


#include "ReleaseType.hpp"


class ReleaseType_continuous : public ReleaseType
{
private:
  // note that this also inherits data members ParticleReleaseType m_rType, int m_parPerTimestep, double m_releaseStartTime,
  //  double m_releaseEndTime, and int m_numPar from ReleaseType.
  // guidelines for how to set these variables within an inherited ReleaseType are given in ReleaseType.hpp.

  int parPerTimestep;


protected:
public:
  // Default constructor
  ReleaseType_continuous()
  {
  }

  // destructor
  ~ReleaseType_continuous()
  {
  }


  virtual void parseValues()
  {
    parReleaseType = ParticleReleaseType::continuous;

    parsePrimitive<int>(true, parPerTimestep, "parPerTimestep");
  }


  void calcReleaseInfo(const double &timestep, const double &simDur)
  {
    // set the overall releaseType variables from the variables found in this class
    m_parPerTimestep = parPerTimestep;
    m_releaseStartTime = 0;
    m_releaseEndTime = simDur;
    int nReleaseTimes = std::ceil(simDur / timestep);
    m_numPar = parPerTimestep * nReleaseTimes;
  }
};
