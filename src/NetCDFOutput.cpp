#include "NetCDFOutput.h"

#include <iostream>

using namespace netCDF;
using namespace netCDF::exceptions;

// constructor, linked to NetCDF file, replace mode only
NetCDFOutput :: NetCDFOutput(std::string output_file) {
    std::cout<< "[NetCDFOutput] \t Writing to " << output_file <<std::endl;
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

// 1D -> int
void NetCDFOutput :: saveField1D(std::string name, const std::vector<size_t> index,
                           int* data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(index, data);
    outfile->sync();
}

// 1D -> float
void NetCDFOutput :: saveField1D(std::string name, const std::vector<size_t> index,
                           float* data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(index, data);
    outfile->sync();
}

// 1D -> double
void NetCDFOutput :: saveField1D(std::string name, const std::vector<size_t> index,
                           double* data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(index, data);
    outfile->sync();
}

// 2D -> int
void NetCDFOutput :: saveField2D(std::string name, std::vector<int>& data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(&data[0]);
    outfile->sync();
}

// 2D -> float
void NetCDFOutput :: saveField2D(std::string name, std::vector<float>& data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(&data[0]);
    outfile->sync();
}

// 2D -> double
void NetCDFOutput :: saveField2D(std::string name, std::vector<double>& data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(&data[0]);
    outfile->sync();
 }

// *D -> int
void NetCDFOutput :: saveField2D(std::string name, const std::vector<size_t> index,
                           std::vector<size_t> size, std::vector<int>& data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(index, size, &data[0]);
    outfile->sync();
}

// *D -> float
void NetCDFOutput :: saveField2D(std::string name, const std::vector<size_t> index,
				 std::vector<size_t> size, std::vector<float>& data) {
  
  // write output data
  NcVar var = fields[name];
  var.putVar(index, size, &data[0]);
  outfile->sync();
}

// *D -> double
void NetCDFOutput :: saveField2D(std::string name, const std::vector<size_t> index,
                           std::vector<size_t> size, std::vector<double>& data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(index, size, &data[0]);
    outfile->sync();
}

// *D -> char
void NetCDFOutput :: saveField2D(std::string name, const std::vector<size_t> index,
                           std::vector<size_t> size, std::vector<char>& data) {
    
    // write output data
    NcVar var = fields[name];
    var.putVar(index, size, &data[0]);
    outfile->sync();
}
