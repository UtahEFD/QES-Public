#ifndef __QUICDATAFILE_QPTURBULENCEFIELD_H__
#define __QUICDATAFILE_QPTURBULENCEFIELD_H__ 1

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <list>

#include "QUICDataFile.h"

// //////////////////////////////////////////////////////////////////
// 
// Class for holding the QP_turbfield.inp file
// 
// //////////////////////////////////////////////////////////////////
class qpTurbulenceField : public quicDataFile
{
public:
  qpTurbulenceField();
  qpTurbulenceField(int nx, int ny, int nz);
  ~qpTurbulenceField();

  std::list<std::string> headerStrings;  // header for the QP_turbfield.dat file that was read if using ASCII
  struct turbFieldData {
    float x;
    float y;
    float z;
    
    float sigU;
    float sigV;
    float sigW;
    
    float lz;
    float leff;
    float eps;
    
    float uv;
    float uw;
    float vw;
    
    float ustar;

    void print() 
    { 
      std::cout << x << ' ' << y << ' ' << z << ' ' 
                << sigU << ' ' << sigV << ' ' << sigW << ' ' 
                << lz << ' ' << leff << ' ' << eps << ' ' 
                << uv << ' ' << uw << ' ' << vw; 
    }
  };

  bool readQUICFile(const std::string &filename);
  bool writeQUICFile(const std::string &filename);

private:

  bool m_asciiRead;
  std::ifstream m_turbFile;

  bool readAsASCII(const std::string &file);

  void parseASCIIHeader();
  void parseASCIIData();

  void parseBinaryData();


};

#endif // #define __QUICDATAFILE_QPTURBULENCEFIELD_H__ 1
