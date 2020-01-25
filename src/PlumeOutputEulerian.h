#pragma once

#include <string>

#include "Eulerian.h"
#include "Dispersion.h"
#include "PlumeInputData.hpp"

#include "NetCDFOutputGeneric.h"

/* Specialized output classes derived from URBOutput_Generic for 
   cell center data (used primarly for vizualization)
*/
class PlumeOutputEulerian : public NetCDFOutputGeneric
{
 public:
 PlumeOutputEulerian()
   : NetCDFOutputGeneric()
    {}
  PlumeOutputEulerian(Dispersion*,PlumeInputData*,std::string);
  ~PlumeOutputEulerian()	       
    {}
  
  // output averaged concentration (call called from outside)
  void save(float);
  
 private:
  // count number of particule in sampling box (can only be called by member)
  void boxCount(const Dispersion*);

  /*
    Sampling box for concentration data
  */
  
  // Copy of the input startTime, 
  // Starting time for averaging for concentration sampling
  float sCBoxTime;
  // Copy of the input timeAvg and timeStep
  float timeAvg,timeStep;
  // time of the concentration output
  float avgOutTime;
       
  // Copies of the input nBoxesX, Y, and Z. 
  // Number of boxes to use for the sampling box
  int nBoxesX,nBoxesY,nBoxesZ;    
  
  // Copies of the input parameters: boxBoundsX1, boxBoundsX2, boxBoundsY1, ... . 
  // upper & lower bounds in each direction of the sampling boxes
  float lBndx,lBndy,lBndz,uBndx,uBndy,uBndz;     

  // these are the box sizes in each direction
  float boxSizeX,boxSizeY,boxSizeZ;
  // volume of the sampling boxes (=nBoxesX*nBoxesY*nBoxesZ)
  float volume;      
  // list of x,y, and z points for the concentration sampling box information
  std::vector<float> xBoxCen,yBoxCen,zBoxCen;
  // sampling box particle counter (for average)
  std::vector<float> cBox;
  // concentration values (for output)
  std::vector<float> conc;      
  
  Dispersion* disp;
};
