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

using namespace netCDF;
using namespace netCDF::exceptions;

class Plume {
    
    public:
        
        Plume(Urb*,Dispersion*,PlumeInputData*,Output*);
        void run(Urb*,Turb*,Eulerian*,Dispersion*,PlumeInputData*,Output*);
        void save(Output*);
        
    private:
        
        int numPar,nx,ny,nz,nBoxesX,nBoxesY,nBoxesZ,tStep,numTimeStep,parPerTimestep;
        double boxSizeX,boxSizeY,boxSizeZ,lBndx,lBndy,lBndz,uBndx,uBndy,uBndz,tStepInp,avgTime,volume;
        double quanX,quanY,quanZ,sCBoxTime;
        std::vector<double> cBox,conc,tStrt,timeStepStamp,xBoxCen,yBoxCen,zBoxCen;
        
        int loopExt=0;
        
        void average(const int, const Dispersion*, const Urb*);
        void outputConc();
        void reflection(double&, double&, const double&, const  double&,  const double&, const double&
        		,double&,double&,const Eulerian*,const Urb*,const int&,const int&,const int&,double&,double&);
        double dot(const pos&, const pos&);
        pos normalize(const pos&);
        pos VecScalarMult(const pos&,const double&);
        pos reflect(const pos&,const pos&);
        pos posSubs(const pos&,const pos&);
        pos posAdd(const pos&,const pos&);
        double distance(const pos&,const pos&);
        
        double min(double[],int);
        double max(double[],int);
        
        // output manager
        int output_counter=0;
        double timeOut=0;
        std::vector<NcDim> dim_scalar_t;
        std::vector<NcDim> dim_scalar_z;
        std::vector<NcDim> dim_scalar_y;
        std::vector<NcDim> dim_scalar_x;
        std::vector<NcDim> dim_vector;
        
        struct AttScalarDbl {
            double* data;
            std::string name;
            std::string long_name;
            std::string units;
            std::vector<NcDim> dimensions;
        };
        
        struct AttVectorDbl {
            std::vector<double>* data;
            std::string name;
            std::string long_name;
            std::string units;
            std::vector<NcDim> dimensions;
        };
        std::map<std::string,AttScalarDbl> map_att_scalar_dbl;
        std::map<std::string,AttVectorDbl> map_att_vector_dbl; 
        std::vector<AttScalarDbl> output_scalar_dbl;       
        std::vector<AttVectorDbl> output_vector_dbl;
};
#endif