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
  
  std::string quicVersion;

 // METInputType metInputFlag;       
  int metInputFlag;            /// QUIC =0 ITT_MM5 =1 HOTMAC =2;
  int numMeasuringSites;
  int maxSizeProfiles;

    // This really needs to be an array of sensors...

  std::string siteName;
  std::string sensorFileName;

  quSensorParams quSensorData;

  quMetParams();
  ~quMetParams() {}
  quMetParams(const quMetParams& other)
    {
      std::cerr<<"Copy constructor called"<<std::endl;
      *this = other;

    }

  //overloaded assignment
  quMetParams& operator= (const quMetParams& other);


  bool readQUICFile(const std::string &filename);
  bool writeQUICFile(const std::string &filename);
  void build_map(); 
private:
};

#endif // #define __QUICDATA_QUMETPARAMS_H__ 1
