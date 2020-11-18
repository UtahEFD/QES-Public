//
//  Fire.hpp
//  
//  This class models fire spread rate using Balbi (2019)
//
//  Created by Matthew Moody, Jeremy Gibbs on 12/27/18.
//

#ifndef FIRE_H
#define FIRE_H

#include "URBInputData.h"
#include "URBGeneralData.h"
#include "FuelProperties.hpp"
#include "Vector3.h"
#include "Solver.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <netcdf>

using namespace netCDF;
using namespace netCDF::exceptions;

class FuelProperties;

class Fire {
    
    
    public:
    
    //Fire(URBInputData*, URBGeneralData*, Output*); FM OBSOLETE
    Fire(URBInputData*, URBGeneralData*);
        
        struct FireProperties {
	  float  w, h, d, r, T, tau, K, H0, U_c, L_c;
        };
        
        struct FireState {
            float burn_time;
            float burn_flag;
            float front_flag;
        };
        
        struct FireCell {
            FireProperties properties;
            FuelProperties* fuel;
            FireState state;
        };
        
        double time=0;
        float r_max = 0;
        float dt=0;
	int FFII_flag=0;
        
        std::vector< FireCell > fire_cells;
        
        std::vector<float> w_base;

        std::vector<float> front_map;

        std::vector<float> del_plus;

        std::vector<float> del_min;

        std::vector<float> xNorm;

        std::vector<float> yNorm;

        std::vector<float> Force;
  
        std::vector<float> Pot_u;
        std::vector<float> Pot_v;
        std::vector<float> Pot_w;

        // output fields
        std::vector<float> burn_flag;
        std::vector<float> burn_out;
	std::vector<float> Pot_w_out;
	

  // Potential field
        int pot_z, pot_r, pot_G, pot_rStar, pot_zStar;
        float drStar, dzStar;
        std::vector<float> u_r;
        std::vector<float> u_z;
        std::vector<float> G;
        std::vector<float> Gprime;
        std::vector<float> rStar;
        std::vector<float> zStar;

        // Fire Arrival Times from netCDF
	int SFT_time, SFT_x1, SFT_y1, SFT_x2, SFT_y2;
	std::vector<float> FT_time;
	std::vector<float> FT_x1;
	std::vector<float> FT_y1;
	std::vector<float> FT_x2;
	std::vector<float> FT_y2;
      
        void run(Solver*, URBGeneralData*);
        void move(Solver*, URBGeneralData*); 
	void potential(URBGeneralData*);   
    //void save(Output*); FM OBSOLETE
        float computeTimeStep();
        
    private:
        
        // grid information
        int nx,ny,nz;
        float dx, dy, dz;
        
        // fire information
        int fuel_type;
        float x_start, y_start;
        float L, W, H, baseHeight, courant;
        int i_start, i_end, j_start, j_end, k_end,k_start;

        float rothermel(FuelProperties*, float, float, float);
                     
        FireProperties balbi(FuelProperties*, float, float, float, float, float, float);
        
        FireProperties runFire(float, float, int);
        
	
    // FM OBSOLETE
        /// Declaration of output manager
	/*
        int output_counter=0;
        std::vector<NcDim> dim_scalar_1,dim_scalar_3,dim_scalar_4;
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
       */
};

#endif
