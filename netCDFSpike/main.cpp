#include <iostream>
#include <vector>

#include <netcdf>

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

// We are writing 2D data, a 6 x 12 grid. 
static const int NX = 6;
static const int NY = 12;

// Return this in event of a problem.
static const int NC_ERR = 2;



int main()
{

    NcFile outFile("lsm.nc", NcFile::replace);
    
    // define dimensions
    NcDim t_dim = outFile.addDim("t");
    NcDim z_dim = outFile.addDim("z",4);
    return 0;

 /* // This is the data array we will write. It will just be filled
  // with a progression of numbers for this example.
  int dataOut[NX][NY];
  
  // Create some pretend data. If this wasn't an example program, we
  // would have some real data to write, for example, model output.
  for(int i = 0; i < NX; i++)
    for(int j = 0; j < NY; j++)
      dataOut[i][j] = i * NY + j;
  
  // The default behavior of the C++ API is to throw an exception i
  // an error occurs. A try catch block is necessary.
   
  try
    {  
      // Create the file. The Replace parameter tells netCDF to overwrite
      // this file, if it already exists.
      NcFile dataFile("simple_xy.nc", NcFile::replace);
      
      // Create netCDF dimensions
      NcDim xDim = dataFile.addDim("x", NX);
      NcDim yDim = dataFile.addDim("y", NY);
      
      // Define the variable. The type of the variable in this case is
      // ncInt (32-bit integer).
      vector<NcDim> dims;
      dims.push_back(xDim);
      dims.push_back(yDim);
      NcVar data = dataFile.addVar("data", ncInt, dims);
   
      // Write the data to the file. Although netCDF supports
      // reading and writing subsets of data, in this case we write all
      // the data in one operation.
      data.putVar(dataOut);
      
      // The file will be automatically close when the NcFile object goes
      // out of scope. This frees up any internal netCDF resources
      // associated with the file, and flushes any buffers.
      
      //cout << "*** SUCCESS writing example file simple_xy.nc!" << endl;
      return 0; 
    }
  catch(NcException& e)
    {e.what();
      return NC_ERR;
    }*/
}


/*
//
//  run_lsm.cpp
//  
//
//  Created by Jeremy Gibbs on 10/30/17
//
#include "input.hpp"
#include "utah_lsm.hpp"
#include "constants.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <netcdf>

using namespace netCDF;
using namespace netCDF::exceptions;

namespace {
    namespace c = Constants;
}

int main () {
    
    // declare local variables
    bool first = true;
    int n_error = 0;
    double utc, atm_ws, net_r;
    double zeta_m=0,zeta_s=0,zeta_o=0,zeta_t=0;
    double ustar,flux_wT,flux_wq,flux_gr;
    NcVar t_var, z_var, ustar_var;
    NcVar flux_wT_var, flux_wq_var, Rnet_var;
    NcVar soil_T_var, soil_q_var, flux_gr_var;
    
    // namelist time section
    double dt, utc_start;
    int nsteps;
    
    // namelist space section
    double z_o, z_t, z_m, z_s;
    int nsoilz;
    
    //namelist pressure section
    double atm_p;
    
    // namelist radiation section
    double albedo, emissivity, latitude, longitude;
    int julian_day, comp_rad;
        
    // print a nice little welcome message
    std::cout << std::endl;
    std::cout<<"##############################################################"<<std::endl;
    std::cout<<"#                                                            #"<<std::endl;
    std::cout<<"#                     Welcome to UtahLSM                     #"<<std::endl;
    std::cout<<"#                                                            #"<<std::endl;
    std::cout<<"#   A land surface model created at the University of Utah   #"<<std::endl;
    std::cout<<"#                                                            #"<<std::endl;
    std::cout<<"##############################################################"<<std::endl;
    
    // initialize the input class
    Input input;
    
    // grab values from length section
    std::cout<<"Processing namelist.ini" << std::endl;
    n_error += input.getItem(&dt,        "time", "dt",        "");
    n_error += input.getItem(&utc_start, "time", "utc_start", "");
    n_error += input.getItem(&nsteps,    "time", "nsteps",    "");
    
    // grab values from pressure section
    n_error += input.getItem(&atm_p, "pressure", "p_o", "");
    
    // grab values from space section
    n_error += input.getItem(&z_o,    "length", "z_o",    "");
    n_error += input.getItem(&z_t,    "length", "z_t",    "");
    n_error += input.getItem(&z_m,    "length", "z_m",    "");
    n_error += input.getItem(&z_s,    "length", "z_s",    "");
    n_error += input.getItem(&nsoilz, "length", "nsoilz", "");
    
    // grab values from radiation section
    n_error += input.getItem(&albedo,     "radiation", "albedo",     "");
    n_error += input.getItem(&emissivity, "radiation", "emissivity", "");
    n_error += input.getItem(&latitude,   "radiation", "latitude",   "");
    n_error += input.getItem(&longitude,  "radiation", "longitude",  "");
    n_error += input.getItem(&julian_day, "radiation", "julian_day", "");
    n_error += input.getItem(&comp_rad,   "radiation", "comp_rad",   "");
    
    // convert latitude and longitude into radians
    latitude  = latitude * c::pi / 180.0;
    longitude = longitude * c::pi / 180.0; 
    
    if (n_error) throw "There was an error reading the input file";
    
    // read in external soil data
    std::cout<<"Processing inputSoil.dat" << std::endl;
    std::vector<int> soil_type;
    std::vector<double> soil_z;
    std::vector<double> soil_T;
    std::vector<double> soil_q;
    std::vector<double> soil_T_last;
    std::vector<double> soil_q_last;
        
    n_error += input.getProf(&soil_z,   "soil", "soil_z",   nsoilz);
    n_error += input.getProf(&soil_type,"soil", "soil_type",nsoilz);
    n_error += input.getProf(&soil_T,   "soil", "soil_T",   nsoilz);
    n_error += input.getProf(&soil_q,   "soil", "soil_q",   nsoilz);
    
    soil_T_last = soil_T;
    soil_q_last = soil_q;
    
    if (n_error) throw "There was an error reading the input file";
        
    // read in external atmospheric data
    std::cout<<"Processing inputMetr.dat" << std::endl;
    std::vector<double> atm_u;
    std::vector<double> atm_v;
    std::vector<double> atm_T;
    std::vector<double> atm_q;
    std::vector<double> R_net;
    
    n_error += input.getProf(&atm_u, "metr", "atm_u", nsteps);
    n_error += input.getProf(&atm_v, "metr", "atm_v", nsteps);
    n_error += input.getProf(&atm_T, "metr", "atm_T", nsteps);
    n_error += input.getProf(&atm_q, "metr", "atm_q", nsteps);
    if (!comp_rad) n_error += input.getProf(&R_net, "metr", "R_net", nsteps);
    
    // modify soil levels to be negative
    std::transform(soil_z.begin(), soil_z.end(), soil_z.begin(),
          bind2nd(std::multiplies<double>(), -1.0));
    
    if (n_error) throw "There was an error reading the input file";
    
    // create output file
    std::cout<<"##############################################################"<<std::endl;
    std::cout<<"Creating output file"<<std::endl;
    NcFile outFile("lsm.nc", NcFile::replace);
    
    // define dimensions
    NcDim t_dim = outFile.addDim("t");
    NcDim z_dim = outFile.addDim("z",nsoilz);
    
    // define variables
    std::vector<NcDim> dim_vector;
    dim_vector.push_back(t_dim);
    dim_vector.push_back(z_dim);
    
    t_var       = outFile.addVar("time",    ncInt,   t_dim);
    z_var       = outFile.addVar("soil_z",  ncFloat, z_dim);
    ustar_var   = outFile.addVar("ustar",   ncFloat, t_dim);
    flux_wT_var = outFile.addVar("flux_wT", ncFloat, t_dim);
    flux_wq_var = outFile.addVar("flux_wq", ncFloat, t_dim);
    flux_gr_var = outFile.addVar("flux_gr", ncFloat, t_dim);
    Rnet_var    = outFile.addVar("Rnet",    ncFloat, t_dim);
    soil_T_var  = outFile.addVar("soil_T",  ncFloat, dim_vector);
    soil_q_var  = outFile.addVar("soil_q",  ncFloat, dim_vector);
    
    std::cout<<"##############################################################"<<std::endl;
    std::cout<<"Running UtahLSM"<<std::endl;;
    std::cout<<"##############################################################"<<std::endl;
    //nsteps = 2;
    for (int t=0; t<nsteps; ++t) {
        
        // check if first time through
        if (t>0) first = false;
        
        // set time
        utc = utc_start + float(t+1)*dt;
        std::cout<<"\rProcessing time: "<<utc<<std::flush;
        
        // compute wind speed
        atm_ws = sqrt( pow(atm_u[t],2) + pow(atm_v[t],2) );
        
        // check whether radiation model is needed
        if (comp_rad) {
            net_r = 0;
        } else {
            net_r = R_net[t];
        }
        
        // Call the model
        UtahLSM utahlsm(first,dt,z_o,z_t,z_m,z_s,
                        atm_p,atm_ws,atm_T[t],atm_q[t],
                        nsoilz,soil_z,soil_type,soil_T,
                        soil_T_last,soil_q,soil_q_last,
                        julian_day,utc,latitude,longitude,
                        albedo,emissivity,net_r,comp_rad,
                        zeta_m,zeta_s,zeta_o,zeta_t,
                        ustar,flux_wT,flux_wq,flux_gr);
        
        // write output data
        const std::vector<size_t> index = {t};
        const std::vector<size_t> time_height_index = {static_cast<size_t>(t), 0};
        std::vector<size_t> time_height_size  = {1, nsoilz};
        t_var.putVar(index, utc);
        z_var.putVar(&soil_z[0]);
        ustar_var.putVar(index, ustar);
        flux_wT_var.putVar(index, c::rho_air*c::Cp_air*flux_wT);
        flux_wq_var.putVar(index, c::rho_air*c::Lv*flux_wq);
        flux_gr_var.putVar(index, flux_gr);
        Rnet_var.putVar(index, net_r);
        soil_T_var.putVar(time_height_index, time_height_size, &soil_T[0]);
        soil_q_var.putVar(time_height_index, time_height_size, &soil_q[0]);
    }
    std::cout<<std::endl;
    std::cout<<"Finished!"<<std::endl;
    std::cout<<"##############################################################"<<std::endl;
    return 0;
}

*/