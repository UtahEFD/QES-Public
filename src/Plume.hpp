//
//  Plume.hpp
//  
//  This class handles plume model
//

#ifndef PLUME_H
#define PLUME_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include "Output.hpp"
#include "Urb.hpp"
#include "Turb.hpp"
#include "Eulerian.h"
#include "Dispersion.h"
#include "TypeDefs.hpp"
#include "PlumeInputData.hpp"

class Plume {
    
    public:
        
        Plume(Urb*,Dispersion*,PlumeInputData*);
        void run(Urb*,Turb*,Eulerian*,Dispersion*,PlumeInputData*,Output*);
        
    private:
        
        int numPar,nx,ny,nz,nBoxesX,nBoxesY,nBoxesZ,tStep,numTimeStep,parPerTimestep;
        double boxSizeX,boxSizeY,boxSizeZ,lBndx,lBndy,lBndz,uBndx,uBndy,uBndz,tStepInp,avgTime,volume;
        double quanX,quanY,quanZ,sCBoxTime;
        std::vector<double> cBox,tStrt,timeStepStamp,xBoxCen,yBoxCen,zBoxCen;
        
        int loopExt=0;
        
        void average(const int, const Dispersion*);
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