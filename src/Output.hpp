//
//  output.hpp
//
//  This class handles netcdf output
//
//  Created by Jeremy Gibbs on 12/19/18.
//

#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include <string>
#include <vector>
#include <map>
#include <netcdf>

/**
 * This class handles saving output files.
 */

using namespace netCDF;
using namespace netCDF::exceptions;

class Output {

    private:
        // netCDF variables
        NcFile* outfile;
        std::map<std::string,NcVar> fields;

    public:

        // initializer
        Output(std::string);

        // setter
        NcDim addDimension(std::string, int size=0);
        NcDim getDimension(std::string);
        void addField(std::string, std::string, std::string, std::vector<NcDim>, NcType);
        void saveField1D(std::string, const std::vector<size_t>, float*);
        void saveField2D(std::string, std::vector<float>&);
        void saveField2D(std::string, const std::vector<size_t>,
                         std::vector<size_t>, std::vector<float>&);
        void saveField2D(std::string, const std::vector<size_t>,
                         std::vector<size_t>, std::vector<int>&);
};

#endif
