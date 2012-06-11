#ifndef __QUICDATAFILE_QUMETPARAMS_H__
#define __QUICDATAFILE_QUMETPARAMS_H__ 1

#include <cstdlib>
#include <fstream>
#include "QUICDataFile.h"
#include "legacyFileParser.h"
#include "QUSensor.h"

// //////////////////////////////////////////////////////////////////
// 
// Class for holding the QU_metparams.inp file
// 
// //////////////////////////////////////////////////////////////////
class quMetParams : public quicDataFile
{
public:
  enum METInputType {
    QUIC = 0,
    ITT_MM5 = 1,
    HOTMAC = 2
  };
  
  METInputType metInputFlag;
  int numMeasuringSites;
  int maxSizeProfiles;

  std::string siteName;
  std::string sensorFileName;

  quSensorParams quSensorData;

  quMetParams();
  ~quMetParams() {}

  bool readQUICFile(const std::string &filename);
  bool writeQUICFile(const std::string &filename);

private:
};

#endif // #define __QUICDATA_QUMETPARAMS_H__ 1
