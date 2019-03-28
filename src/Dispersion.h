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
#include "Eulerian.h"
#include "Random.h"

#include <helper_math.h>

class dispersion {
    public:
        
        Eulerian eul; 
        void createDisp(const Eulerian&);
        
        struct matrix {
            double x;
            double y;
            double z;
        };
  
        std::vector<float3> pos,prime;
        std::vector<double> zIniPos,wPrime,timeStepStamp,tStrt;
        double eps;
        int numTimeStep, parPerTimestep;
        
    private:
        
        double xSrc,ySrc,zSrc;
        int numPar;
};
#endif
