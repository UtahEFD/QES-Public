//
//  Input.hpp
//  
//  This class handles netcdf input
//
//  Created by Jeremy Gibbs on 03/15/19.
//

#ifndef INPUT_HPP
#define INPUT_HPP

#include <string>
#include <vector>
#include <map>
#include <netcdf>

/**
 * This class handles reading input files.
 */

using namespace netCDF;
using namespace netCDF::exceptions;

class Input {
    
    private:
        // netCDF variables
        NcFile* infile;
    
    public:
    
        // initializer
        Input(std::string);
        
        // getters
        void getDimension(std::string, NcDim&);
        void getDimensionSize(std::string name, int&);
        void getVariable(std::string, NcVar&);
        void getVariableData(std::string name,std::vector<double>&);
        void getVariableData(std::string name,std::vector<size_t>,std::vector<size_t>,std::vector<double>&);
};

#endif
