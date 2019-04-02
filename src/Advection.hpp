//
//  Advection.hpp
//  
//  This class handles advection of particles
//

#ifndef ADVECTION_H
#define ADVECTION_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include "Urb.hpp"
#include "Turb.hpp"
#include "Eulerian.h"
#include "Dispersion.h"
#include "TypeDefs.hpp"
#include "PlumeInputData.hpp"

class Advection {
    
    public:
        
        Advection(Urb*,Turb*,Eulerian*,Dispersion*);
        
    private:
        
        int numPar,nx,ny,nz,numBoxX,numBoxY,numBoxZ,tStep;
        double xBoxSize,yBoxSize,zBoxSize,lBndx,lBndy,lBndz,uBndx,uBndy,uBndz,tStepInp,avgTime,volume;
        std::vector<double> cBox,tStrt,timeStepStamp,xBoxCen,yBoxCen,zBoxCen;
        
        std::ofstream output;
        std::ofstream rand_output;
        int loopExt=0;
        
        void average(const int, const Dispersion&);
        void outputConc();
        void reflection(double&, double&, const double&, const  double&,  const double&, const double&
        		,double&,double&,const Eulerian&,const int&,const int&,const int&,double&,double&);
        double dot(const pos&, const pos&);
        pos normalize(const pos&);
        pos VecScalarMult(const pos&,const double&);
        pos reflect(const pos&,const pos&);
        pos posSubs(const pos&,const pos&);
        pos posAdd(const pos&,const pos&);
        double distance(const pos&,const pos&);
        
        double min(double[],int);
        double max(double[],int);
};
#endif