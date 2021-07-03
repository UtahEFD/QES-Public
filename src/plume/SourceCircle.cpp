
#include "SourceCircle.hpp"


void SourceCircle::checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend)
{
  if (radius < 0) {
    std::cerr << "ERROR (SourceCircle::checkPosInfo): input radius is negative! radius = \"" << radius << "\"" << std::endl;
    exit(1);
  }

  if ((posX - radius) < domainXstart || (posX + radius) > domainXend) {
    std::cerr << "ERROR (SourceCircle::checkPosInfo): input posX+radius is outside of domain! posX = \"" << posX << "\" radius = \"" << radius
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if ((posY - radius) < domainYstart || (posY + radius) > domainYend) {
    std::cerr << "ERROR (SourceCircle::checkPosInfo): input posY+radius is outside of domain! posY = \"" << posY << "\" radius = \"" << radius
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if ((posZ - radius) < domainZstart || (posZ + radius) > domainZend) {
    std::cerr << "ERROR (SourceCircle::checkPosInfo): input posZ is outside of domain! posZ = \"" << posZ << "\" radius = \"" << radius
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }
}


int SourceCircle::emitParticles(const float dt, const float currTime, std::list<Particle *> &emittedParticles)
{
  // warning!!! this is still a point source! Need to work out the geometry details still
  // release particle per timestep only if currTime is between m_releaseStartTime and m_releaseEndTime
  if (currTime >= m_rType->m_releaseStartTime && currTime <= m_rType->m_releaseEndTime) {

    for (int pidx = 0; pidx < m_rType->m_parPerTimestep; pidx++) {

      Particle *cPar = new Particle();

      cPar->xPos_init = posX;
      cPar->yPos_init = posY;
      cPar->zPos_init = posZ;

      cPar->tStrt = currTime;

      cPar->sourceIdx = sourceIdx;

      emittedParticles.push_front(cPar);
    }
  }

  return emittedParticles.size();//m_rType->m_parPerTimestep;
}
