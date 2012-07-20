#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <vector>
#include <algorithm>
class util{
 public:
  util();
  void readInputFile();
  int twidth,numPar, nx,ny,nz, theight, windFieldData,numBoxX,numBoxY,numBoxZ,numBuild,ibuild;
  double timeStep,ustar,dur,sCBoxTime,eCBoxTime, avgTime;
  double xSrc,ySrc,zSrc,rSrc,xBoxSize,yBoxSize,zBoxSize,zo,dx,dy,dz,vonKar;
  double bnds[6];
  std::string file,src;
  std::vector<double> xfo,yfo,zfo,hgt,wth,len;
 private:
  int profile;
  std::string outFile;
};
#endif
