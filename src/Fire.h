// QUICFire spread rate using Balbi (2009) model
// Matthew Moody
// Dec 27, 2018

#ifndef FIRE_H
#define FIRE_H

#include "URBInputData.h"
#include "FuelProperties.hpp"
#include "Vector3.h"
#include "Output.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>

class FuelProperties;

class Fire {
    
    
    public:
        
        Fire(URBInputData*, Output*);
        
        struct FireProperties {
            double  w, h, rx, ry, T, tau;
        };
        
        struct FireState {
            double burn_time;
            double burn_flag;
        };
        
        struct FireCell {
            FireProperties properties;
            FuelProperties* fuel;
            FireState state;
        };
        
        std::vector< FireCell > allFireCells;
        
        /**
         * @brief 
         * 
         * This function saves user-defined data to file
         */
         void save(Output*);
        
    private:
        
        // grid information
        int nx,ny,nz;
        float dx, dy, dz;
        
        // fire information
        int fuel_type;
        float x_start, y_start;
        float L, W, H, baseHeight;
        int i_start, i_end, j_start, j_end, k_end,k_start;
        
        double Rothermel(double, double, double, double, double, 
                         double, double, double, double, double, 
                         double, double, double, double, double, 
                         double, double, double, double);
                     
        FireProperties Balbi(double, double, double, double, double,
                             double, double, double, double, double);
        
        FireProperties runFire(double, double, int);
        
        
        /// Declaration of output manager
        int output_counter=0;
        double time=0;
        std::vector<NcDim> dim_scalar_t;
        std::vector<std::string> output_fields;
        
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
        
        struct AttVectorInt {
            std::vector<int>* data;
            std::string name;
            std::string long_name;
            std::string units;
            std::vector<NcDim> dimensions;
        };
        
        std::map<std::string,AttScalarDbl> map_att_scalar_dbl;
        std::map<std::string,AttVectorDbl> map_att_vector_dbl;
        std::map<std::string,AttVectorInt> map_att_vector_int;
        
        std::vector<AttScalarDbl> output_scalar_dbl;
        std::vector<AttVectorDbl> output_vector_dbl;
        std::vector<AttVectorInt> output_vector_int;
};

#endif