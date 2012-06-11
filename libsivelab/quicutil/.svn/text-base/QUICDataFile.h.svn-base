#ifndef __QUICDATAFILE_H__
#define __QUICDATAFILE_H__ 1

#include <string>

class quicDataFile
{
public:
  quicDataFile()
  : beVerbose(true)
  {}
  
  virtual ~quicDataFile() {}
  
  virtual bool readQUICFile(const std::string &filename) = 0;
  virtual bool writeQUICFile(const std::string &filename) = 0;

  bool beVerbose;

protected:
private:
};

#endif // #ifndef __QUICDATAFILE_H__
