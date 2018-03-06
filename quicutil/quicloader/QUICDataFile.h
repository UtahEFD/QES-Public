#ifndef __QUICDATAFILE_H__
#define __QUICDATAFILE_H__ 1
#include "language_map.h"
#include <string>

class quicDataFile :public languageMap        ////languageMap
{
public:
  quicDataFile()
    : beVerbose(true), quicVersionString("")
  {}

  quicDataFile( const std::string &version )
    : beVerbose(true), quicVersionString( version )
  {}
  
  virtual ~quicDataFile() {}
  
  virtual bool readQUICFile(const std::string &filename) = 0;
  virtual bool writeQUICFile(const std::string &filename) = 0;

  void setQUICVersionString( const std::string &version ) { quicVersionString = version; }
  std::string getQUICVersionString() { return quicVersionString; }

  bool beVerbose;

protected:

  std::string quicVersionString;

private:
};

#endif // #ifndef __QUICDATAFILE_H__
