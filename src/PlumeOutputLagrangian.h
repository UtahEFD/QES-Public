
#pragma once


#include <string>


#include "PlumeInputData.hpp"
#include "Eulerian.h"
#include "Dispersion.h"


#include "NetCDFOutputGeneric.h"


/* Specialized output classes derived from URBOutput_Generic for 
   cell center data (used primarly for vizualization)
*/
class PlumeOutputLagrangian : public NetCDFOutputGeneric
{
 public:
 PlumeOutputLagrangian()
   : NetCDFOutputGeneric()
    {}
  PlumeOutputLagrangian(Dispersion*,PlumeInputData*,std::string);
  ~PlumeOutputLagrangian()	       
    {}
  
  // output averaged concentration (call called from outside)
  void save(float);
  
 private:
  
   /*
     Output times 
   */ 

  // FM -> meed ot create dedicated input variables
  // time to start of Lag output
  float stLagTime_;
  // time to output (1st output -> updated each time)
  float outLagTime_;
  // Copy of the input timeAvg and timeStep 
  float outLagFreq_;

  /*
    Particle data
  */
  
  // total number of particle to be released 
  int numPar_;
  
  // list of particles ID (for NetCDF dimension)
  std::vector<float> ParID;
  
  // list of x,y,z position for particles
  std::vector<float> xPos,yPos,zPos;
  
  Dispersion* disp_;
};
