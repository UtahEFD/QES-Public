#ifndef __QUICDATAFILE_QPSOURCE_H__
#define __QUICDATAFILE_QPSOURCE_H__ 1

#include <iostream>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>

#include "QUICDataFile.h"

class qpSource : public quicDataFile
{
public:
  qpSource()
  : quicDataFile()
  {}
  
  ~qpSource() {}

  bool readQUICFile(const std::string &filename);
  bool writeQUICFile(const std::string &filename);

  // TODO Better enum names?
  enum StrengthUnitType
  {
    G = 1,
    G_PER_S = 2,
    L = 3,
    L_PER_S = 4
  };
  
  enum ReleaseType
  {
    INSTANTANEOUS = 1,
    CONTINUOUS = 2,
    DISCRETE_CONTINUOUS = 3
  };
  
  enum SourceGeometryType
  {
    SPHERICAL_SHELL = 1,
    LINE = 2,
    CYLINDER = 3,
    EXPLOSIVE = 4,
    AREA = 5,
    MOVING_POINT = 6,
    SPHERICAL_VOLUME = 7,
    SUBMUNITIONS = 8
  };

  struct DataPoint
  {
    float x, y, z;
  };

  struct SourceInfo
  {
    std::string name;
    StrengthUnitType strengthUnits;  // Source strength units (1 = g, 2 = g/s, 3 = L,4 = L/s)
    float strength;     // Source Strength
    float density; // Source Density (kg/m^3) [Only used for Volume based source strengths]
    
    ReleaseType release;

    float startTime;
    float duration;

    SourceGeometryType geometry;
    std::vector<DataPoint> points;

    float radius;
  };

  int numberOfSources;
  int numberOfSourceNodes;

  std::vector<SourceInfo> sources;

private:
};

#endif // #define __QUICDATA_QPSOURCE_H__ 1
