#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <vector>
#include <algorithm>

#include "quicloader/QUICProject.h"

class util{
 public:
  util();
    void readInputFile(const std::string &quicFileToLoad);
  int twidth,numPar, nx,ny,nz, theight, windFieldData,numBoxX,numBoxY,numBoxZ,numBuild,ibuild;
  double timeStep,ustar,dur,sCBoxTime,eCBoxTime, avgTime;
  double xSrc,ySrc,zSrc,rSrc,xBoxSize,yBoxSize,zBoxSize,zo,dx,dy,dz,vonKar;
  double bnds[6];
  std::string file,src;
  std::vector<double> xfo,yfo,zfo,hgt,wth,len;

    sivelab::QUICProject m_QUICProjData;
    
 private:
  int profile;
  std::string outFile;
};
#endif
