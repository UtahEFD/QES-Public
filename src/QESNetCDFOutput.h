/*
 * QES-Winds
 *
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 *
 */


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
