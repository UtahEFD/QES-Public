//
//  Dispersion.h
//  
//  This class handles dispersion information
//

#ifndef DISPERSION_H
#define DISPERSION_H

#include <list>
#include <vector>
#include <iostream>
#include <cmath>
#include "PlumeInputData.hpp"
#include "Eulerian.h"
#include "Random.h"

#include <helper_math.h>

class Dispersion {
    
    public:
        
        Dispersion(Urb*,Turb*,PlumeInputData*);
                
        struct matrix {
            double x;
            double y;
            double z;
        };
  
        std::vector<float3> pos, prime;
        std::vector<double> zIniPos,wPrime,timeStepStamp,tStrt;
        double eps;
        int numTimeStep, parPerTimestep;
        
    private:
        
        double dur,srcX,srcY,srcZ;
        int dt,nx,ny,numPar;
};
#endif
