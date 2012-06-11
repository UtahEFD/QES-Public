#ifndef __QUICDATAFILE_QUFILEOPTIONS_H__
#define __QUICDATAFILE_QUFILEOPTIONS_H__ 1

#include "QUICDataFile.h"
#include "legacyFileParser.h"


class quFileOptions : public quicDataFile
{
public:
  quFileOptions();
  ~quFileOptions() {}
  
  bool readQUICFile(const std::string &filename);
  bool writeQUICFile(const std::string &filename);
  
  enum FILE_FORMAT_TYPE {
    ASCII  = 1,
    BINARY = 2,
    BOTH   = 3
  };
  
  FILE_FORMAT_TYPE format_type;
  
  bool uofield_flag;
  bool uosensor_flag;
  bool staggered_flag;
  
private:
};

#endif
