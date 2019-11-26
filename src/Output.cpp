//
//  output.hpp
//
//  This class handles netcdf output
//
//  Created by Jeremy Gibbs on 12/19/18.
//

#include "Output.hpp"
#include <iostream>

using namespace netCDF;
using namespace netCDF::exceptions;

Output :: Output(std::string output_file) {

    outfile = new NcFile(output_file, NcFile::replace);
}

NcDim Output :: addDimension(std::string name, int size) {

    if (size) {
        return outfile->addDim(name, size);
    } else {
        return outfile->addDim(name);
    }
}

NcDim Output :: getDimension(std::string name) {

    return outfile->getDim(name);
}

void Output :: addField(std::string name, std::string units, std::string long_name,
                        std::vector<NcDim> dims, NcType type) {

    NcVar var;

    var = outfile->addVar(name, type, dims);
    var.putAtt("units", units);
    var.putAtt("long_name", long_name);
    fields[name] = var;
}

void Output :: saveField1D(std::string name, const std::vector<size_t> index,
                           float* data) {

    // write output data
    NcVar var = fields[name];
    var.putVar(index, data);
    outfile->sync();
}

void Output :: saveField2D(std::string name, const std::vector<size_t> index,
                           std::vector<size_t> size, std::vector<float>& data) {

    // write output data
    NcVar var = fields[name];
    var.putVar(index, size, &data[0]);
    outfile->sync();
}

void Output :: saveField2D(std::string name, const std::vector<size_t> index,
                           std::vector<size_t> size, std::vector<int>& data) {

    // write output data
    NcVar var = fields[name];
    var.putVar(index, size, &data[0]);
    outfile->sync();
}

void Output :: saveField2D(std::string name, std::vector<float>& data) {

    // write output data
    NcVar var = fields[name];
    var.putVar(&data[0]);
    outfile->sync();
}
