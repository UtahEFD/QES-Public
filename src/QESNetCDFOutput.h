#pragma once

#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <netcdf>

#include "NetCDFOutput.h"

/*
  This class handles saving output files.

  Attribute are create based and the type of the data
  -> attribute are store in map_att_*
  -> all possible attribute available for the derived class should be
  created by its constructor.
  -> attribute are pushed back to output_* based on what is selected
  by output_fields
  -> the methods allows to by type generic (as long as the data is
  either int,float or double
*/

using namespace netCDF;
using namespace netCDF::exceptions;

//Attribute for scalar/vector for each type
struct AttScalarInt {
    int* data;
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

struct AttScalarFlt {
    float* data;
    std::string name;
    std::string long_name;
    std::string units;
    std::vector<NcDim> dimensions;
};
struct AttVectorFlt {
    std::vector<float>* data;
    std::string name;
    std::string long_name;
    std::string units;
    std::vector<NcDim> dimensions;
};

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

class QESNetCDFOutput : public NetCDFOutput
{
public:
    QESNetCDFOutput()
    {}
    QESNetCDFOutput(std::string);
    virtual ~QESNetCDFOutput()
    {}

    //save function be call outside
    virtual void save(float) = 0;

protected:

    // create attribute scalar based on type of data
    void createAttScalar(std::string,std::string,std::string,
                         std::vector<NcDim>,int*);
    void createAttScalar(std::string,std::string,std::string,
                         std::vector<NcDim>,float*);
    void createAttScalar(std::string,std::string,std::string,
                         std::vector<NcDim>,double*);

    // create attribute vector based on type of data
    void createAttVector(std::string,std::string,std::string,
                         std::vector<NcDim>,std::vector<int>*);
    void createAttVector(std::string,std::string,std::string,
                         std::vector<NcDim>,std::vector<float>*);
    void createAttVector(std::string,std::string,std::string,
                         std::vector<NcDim>,std::vector<double>*);

    // add fields based on output_fields
    void addOutputFields();
    // removed field
    void rmOutputField(std::string);
    void rmTimeIndepFields();
    // save fields
    void saveOutputFields();

    virtual bool validateFileOtions()
    {
        return true;
    };

    int output_counter=0;
    double time=0;

    /* vector containing fields to add to the NetCDF file
       Note: this vector is used ONLY for creating fields
       (i.e. by the constuctor &add function) NOT to save
       them (i.e. by the function save)
    */
    std::vector<std::string> output_fields;

    /* output fields in the NetCDF file for scalar/vector
       for each type.
       Note: this is used ONLY to create and link fields.
    */
    std::map<std::string,AttScalarInt> map_att_scalar_int;
    std::map<std::string,AttScalarFlt> map_att_scalar_flt;
    std::map<std::string,AttScalarDbl> map_att_scalar_dbl;
    std::map<std::string,AttVectorInt> map_att_vector_int;
    std::map<std::string,AttVectorFlt> map_att_vector_flt;
    std::map<std::string,AttVectorDbl> map_att_vector_dbl;

    /* vectors of output fields in the NetCDF file for
       scalar/vector for each type.
       Note: this is used to save the fields, ONLY the
       fields in these 6 vectors will be saved
    */
    std::vector<AttScalarInt> output_scalar_int;
    std::vector<AttScalarFlt> output_scalar_flt;
    std::vector<AttScalarDbl> output_scalar_dbl;
    std::vector<AttVectorInt> output_vector_int;
    std::vector<AttVectorFlt> output_vector_flt;
    std::vector<AttVectorDbl> output_vector_dbl;

};
