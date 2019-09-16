#pragma once

#include 'NetCDFOutput.h'

#include <iostream>

using namespace netCDF;
using namespace netCDF::exceptions;

NetCDFOutput :: NetCDFOutput(std::string output_file) {
    
    outfile = new NcFile(output_file, NcFile::replace);
}

NcDim NetCDFOutput :: addDimension(std::string name, int size) {
    
    if (size) {
        return outfile->addDim(name, size);
    } else {
        return outfile->addDim(name);
    }
}

NcDim NetCDFOutput :: getDimension(std::string name) {
    
    return outfile->getDim(name);
}

void NetCDFOutput :: addField(std::string name, std::string units, std::string long_name,
                        std::vector<NcDim> dims, NcType type) {
 
    NcVar var;

    var = outfile->addVar(name, type, dims);
    var.putAtt("units", units);
    var.putAtt("long_name", long_name);
    fields[name] = var;
}

void NetCDFOutput :: saveField1D(std::string name, const std::vector<size_t> index,
                           double* data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(index, data);
    outfile->sync();
}

void NetCDFOutput :: saveField2D(std::string name, const std::vector<size_t> index,
                           std::vector<size_t> size, std::vector<double>& data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(index, size, &data[0]);
    outfile->sync();
}

void NetCDFOutput :: saveField2D(std::string name, const std::vector<size_t> index,
                           std::vector<size_t> size, std::vector<int>& data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(index, size, &data[0]);
    outfile->sync();
}

void NetCDFOutput :: saveField2D(std::string name, std::vector<double>& data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(&data[0]);
    outfile->sync();
}
