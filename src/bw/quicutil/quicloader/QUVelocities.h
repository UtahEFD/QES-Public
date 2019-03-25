#ifndef __QUICDATAFILE_QUVELOCITIES_H__
#define __QUICDATAFILE_QUVELOCITIES_H__ 1

#include "QUICDataFile.h"

#include "velocities.h"

class quSimParams;

class quVelocities : public quicDataFile
{
public:
  quVelocities() {}
  ~quVelocities() {}
  
  bool readQUICFile(const std::string &filename);
  bool writeQUICFile(const std::string &filename);
  
  bool writeQUICFile(const std::string &filename, const QUIC::velocities &vlcts, const quSimParams &simParams);
  
  QUIC::velocities vlcts;
  
  int grid_row;
  int grid_slc;

  int nx;
  int ny;
  int nz;
  
  int ndx;
  int ndx_pi;
  int ndx_pj;
  int ndx_pk;
  
private:
};

#endif
