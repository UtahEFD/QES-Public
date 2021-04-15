/****************************************************************************
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
 ****************************************************************************/

/**
 * @file QESNetCDFOutput.cpp
 * @brief Handles the saving of output files.
 * Attributes are created based on the type of the data:
 *   - Attributes are stored in map_att_*
 *   - All possible attributes available for the derived class should be created by its CTOR.
 *   - Attributes are pushed back to output_* based on what is selected by output_fields
 *   - The methods allow to be type generic (as long as the data is either int, float, or double)
 */

#include "QESNetCDFOutput.h"

QESNetCDFOutput::QESNetCDFOutput(std::string output_file)
    : NetCDFOutput(output_file)
{
};

//----------------------------------------
// create attribute scalar
// -> int
void QESNetCDFOutput::createAttScalar(std::string name,
                                      std::string long_name,
                                      std::string units,
                                      std::vector<NcDim> dims,
                                      int* data)
{
    // FM -> here I do not know what is the best way to add the ref to data.
    AttScalarInt att = {data,name,long_name,units,dims};
    map_att_scalar_int.emplace(name,att);


}
// -> float
void QESNetCDFOutput::createAttScalar(std::string name,
                                      std::string long_name,
                                      std::string units,
                                      std::vector<NcDim> dims,
                                      float* data)
{
    // FM -> here I do not know what is the best way to add the ref to data.
    AttScalarFlt att = {data,name,long_name,units,dims};
    map_att_scalar_flt.emplace(name,att);

}
// -> double
void QESNetCDFOutput::createAttScalar(std::string name,
                                      std::string long_name,
                                      std::string units,
                                      std::vector<NcDim> dims,
                                      double* data)
{
    // FM -> here I do not know what is the best way to add the ref to data.
    AttScalarDbl att = {data,name,long_name,units,dims};
    map_att_scalar_dbl.emplace(name,att);
}

//----------------------------------------
// create attribute Vector
// -> int
void QESNetCDFOutput::createAttVector(std::string name,
                                      std::string long_name,
                                      std::string units,
                                      std::vector<NcDim> dims,
                                      std::vector<int>* data)
{
    // FM -> here I do not know what is the best way to add the ref to data.
    AttVectorInt att = {data,name,long_name,units,dims};
    map_att_vector_int.emplace(name,att);

}
// -> float
void QESNetCDFOutput::createAttVector(std::string name,
                                      std::string long_name,
                                      std::string units,
                                      std::vector<NcDim> dims,
                                      std::vector<float>* data)
{
    // FM -> here I do not know what is the best way to add the ref to data.
    AttVectorFlt att = {data,name,long_name,units,dims};
    map_att_vector_flt.emplace(name,att);

}
// -> double
void QESNetCDFOutput::createAttVector(std::string name,
                                      std::string long_name,
                                      std::string units,
                                      std::vector<NcDim> dims,
                                      std::vector<double>* data)
{
    // FM -> here I do not know what is the best way to add the ref to data.
    AttVectorDbl att = {data,name,long_name,units,dims};
    map_att_vector_dbl.emplace(name,att);
}

//----------------------------------------
void QESNetCDFOutput::addOutputFields()
{
    /*
      This function add the  fields to the output vectors
      and link them to the NetCDF.

      Since the type is not know, one needs to loop through
      the 6 output vector to find it.

      FMargairaz
    */

    // create list of fields to save base on output_fields
    for (size_t i=0; i<output_fields.size(); i++) {
        std::string key = output_fields[i];

        if (map_att_scalar_int.count(key)) {
            // scalar int
            output_scalar_int.push_back(map_att_scalar_int[key]);
        } else if (map_att_scalar_flt.count(key)) {
            // scalar flt
            output_scalar_flt.push_back(map_att_scalar_flt[key]);
        } else if (map_att_scalar_dbl.count(key)) {
            // scalar dbl
            output_scalar_dbl.push_back(map_att_scalar_dbl[key]);
        } else if(map_att_vector_int.count(key)) {
            // vector int
            output_vector_int.push_back(map_att_vector_int[key]);
        } else if(map_att_vector_flt.count(key)) {
            // vector flt
            output_vector_flt.push_back(map_att_vector_flt[key]);
        } else if (map_att_vector_dbl.count(key)) {
            // vector dbl
            output_vector_dbl.push_back(map_att_vector_dbl[key]);
        }
    }

    // add scalar fields
    // -> int
    for ( AttScalarInt att : output_scalar_int ) {
        addField(att.name, att.units, att.long_name, att.dimensions, ncInt);
    }
    // -> float
    for ( AttScalarFlt att : output_scalar_flt ) {
        addField(att.name, att.units, att.long_name, att.dimensions, ncFloat);
    }
    // -> double
    for ( AttScalarDbl att : output_scalar_dbl ) {
        addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
    }
    // add vector fields
    // -> int
    for ( AttVectorInt att : output_vector_int ) {
        addField(att.name, att.units, att.long_name, att.dimensions, ncInt);
    }
    // -> int
    for ( AttVectorFlt att : output_vector_flt ) {
        addField(att.name, att.units, att.long_name, att.dimensions, ncFloat);
    }
    // -> double
    for ( AttVectorDbl att : output_vector_dbl ) {
        addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
    }

};



void QESNetCDFOutput::rmOutputField(std::string name)
{
    /*
      This function remove a field from the output vectors
      Since the type is not know, one needs to loop through
      the 6 output vector to find it.

      Note: the field CANNOT be added again.

      FMargairaz
    */

    // loop through scalar fields to remove
    // -> int
    for (unsigned int i=0; i<output_scalar_int.size(); i++) {
        if(output_scalar_int[i].name==name) {
            output_scalar_int.erase(output_scalar_int.begin()+i);
            return;
        }
    }
    // -> float
    for (unsigned int i=0; i<output_scalar_flt.size(); i++) {
        if(output_scalar_flt[i].name==name) {
            output_scalar_flt.erase(output_scalar_flt.begin()+i);
            return;
        }
    }

    // -> double
    for (unsigned int i=0; i<output_scalar_dbl.size(); i++) {
        if(output_scalar_dbl[i].name==name) {
            output_scalar_dbl.erase(output_scalar_dbl.begin()+i);
            return;
        }
    }

    // loop through vector fields to remove
    // -> int
    for (unsigned int i=0; i<output_vector_int.size(); i++) {
        if(output_vector_int[i].name==name) {
            output_vector_int.erase(output_vector_int.begin()+i);
            return;
        }
    }
    // -> float
    for (unsigned int i=0; i<output_vector_flt.size(); i++) {
        if(output_vector_flt[i].name==name) {
            output_vector_flt.erase(output_vector_flt.begin()+i);
            return;
        }
    }
    // -> double
    for (unsigned int i=0; i<output_vector_dbl.size(); i++) {
        if(output_vector_dbl[i].name==name) {
            output_vector_dbl.erase(output_vector_dbl.begin()+i);
            return;
        }
    }
};

void QESNetCDFOutput::rmTimeIndepFields()
{
    /*
      This function remove time indep field from the output vectors
      Since the types are not know, one needs to loop through
      the 6 output vector to find it.

      Note: the fields CANNOT be added again.

      FMargairaz
    */

    // loop through scalar fields to remove
    // -> int
    for (unsigned int i=0; i<output_scalar_int.size(); i++) {
        if(output_scalar_int[i].dimensions[0].getName()!="t") {
            output_scalar_int.erase(output_scalar_int.begin()+i);
        }
    }
    // -> float
    for (unsigned int i=0; i<output_scalar_flt.size(); i++) {
        if(output_scalar_flt[i].dimensions[0].getName()!="t") {
            output_scalar_flt.erase(output_scalar_flt.begin()+i);
        }
    }

    // -> double
    for (unsigned int i=0; i<output_scalar_dbl.size(); i++) {
        if(output_scalar_dbl[i].dimensions[0].getName()!="t") {
            output_scalar_dbl.erase(output_scalar_dbl.begin()+i);
        }
    }

    // loop through vector fields to remove
    // -> int
    for (unsigned int i=0; i<output_vector_int.size(); i++) {
        if(output_vector_int[i].dimensions[0].getName()!="t") {
            output_vector_int.erase(output_vector_int.begin()+i);
        }
    }
    // -> float
    for (unsigned int i=0; i<output_vector_flt.size(); i++) {
        if(output_vector_flt[i].dimensions[0].getName()!="t") {
            output_vector_flt.erase(output_vector_flt.begin()+i);
        }
    }
    // -> double
    for (unsigned int i=0; i<output_vector_dbl.size(); i++) {
        if(output_vector_dbl[i].dimensions[0].getName()!="t") {
            output_vector_dbl.erase(output_vector_dbl.begin()+i);
        }
    }
};

void QESNetCDFOutput::saveOutputFields()
{
    /*
      This function save the fields from the output vectors
      Since the type is not know, one needs to loop through
      the 6 output vector to find it.

      FMargairaz
    */

    // loop through scalar fields to save
    // -> int
    for (unsigned int i=0; i<output_scalar_int.size(); i++) {
        std::vector<size_t> scalar_index;
        scalar_index = {static_cast<unsigned long>(output_counter)};
        saveField1D(output_scalar_int[i].name, scalar_index,
                    output_scalar_int[i].data);
    }
    // -> float
    for (unsigned int i=0; i<output_scalar_flt.size(); i++) {
        std::vector<size_t> scalar_index;
        scalar_index = {static_cast<unsigned long>(output_counter)};
        saveField1D(output_scalar_flt[i].name, scalar_index,
                    output_scalar_flt[i].data);
    }
    // -> double
    for (unsigned int i=0; i<output_scalar_dbl.size(); i++) {
        std::vector<size_t> scalar_index;
        scalar_index = {static_cast<unsigned long>(output_counter)};
        saveField1D(output_scalar_dbl[i].name, scalar_index,
                    output_scalar_dbl[i].data);
    }

    // loop through vector fields to save
    // -> int
    for (unsigned int i=0; i<output_vector_int.size(); i++) {

        std::vector<size_t> vector_index;
        std::vector<size_t> vector_size;

        // if var is time dep -> special treatement for time
        if(output_vector_int[i].dimensions[0].getName()=="t"){
            vector_index.push_back(static_cast<size_t>(output_counter));
            vector_size.push_back(1);
            for(unsigned int d=1;d<output_vector_int[i].dimensions.size();d++){
                int dim=output_vector_int[i].dimensions[d].getSize();
                vector_index.push_back(0);
                vector_size.push_back(static_cast<unsigned long>(dim));
            }
        }
        // if var not time dep -> use direct dimensions
        else{
            for(unsigned int d=0;d<output_vector_int[i].dimensions.size();d++){
                int dim=output_vector_int[i].dimensions[d].getSize();
                vector_index.push_back(0);
                vector_size.push_back(static_cast<unsigned long>(dim));
            }
        }

        saveField2D(output_vector_int[i].name, vector_index,
                    vector_size, *output_vector_int[i].data);
    }
    // -> float
    for (unsigned int i=0; i<output_vector_flt.size(); i++) {
        std::vector<size_t> vector_index;
        std::vector<size_t> vector_size;

        // if var is time dep -> special treatement for time
        if(output_vector_flt[i].dimensions[0].getName()=="t"){
            vector_index.push_back(static_cast<size_t>(output_counter));
            vector_size.push_back(1);
            for(unsigned int d=1;d<output_vector_flt[i].dimensions.size();d++){
                int dim=output_vector_flt[i].dimensions[d].getSize();
                vector_index.push_back(0);
                vector_size.push_back(static_cast<unsigned long>(dim));
            }
        }
        // if var not time dep -> use direct dimensions
        else{
            for(unsigned int d=0;d<output_vector_flt[i].dimensions.size();d++){
                int dim=output_vector_flt[i].dimensions[d].getSize();
                vector_index.push_back(0);
                vector_size.push_back(static_cast<unsigned long>(dim));
            }
        }

        saveField2D(output_vector_flt[i].name, vector_index,
                    vector_size, *output_vector_flt[i].data);
    }
    // -> double
    for (unsigned int i=0; i<output_vector_dbl.size(); i++) {
        std::vector<size_t> vector_index;
        std::vector<size_t> vector_size;

        // if var is time dep -> special treatement for time
        if(output_vector_dbl[i].dimensions[0].getName()=="t"){
            vector_index.push_back(static_cast<size_t>(output_counter));
            vector_size.push_back(1);
            for(unsigned int d=1;d<output_vector_dbl[i].dimensions.size();d++){
                int dim=output_vector_dbl[i].dimensions[d].getSize();
                vector_index.push_back(0);
                vector_size.push_back(static_cast<unsigned long>(dim));
            }
        }
        // if var not time dep -> use direct dimensions
        else{
            for(unsigned int d=0;d<output_vector_dbl[i].dimensions.size();d++){
                int dim=output_vector_dbl[i].dimensions[d].getSize();
                vector_index.push_back(0);
                vector_size.push_back(static_cast<unsigned long>(dim));
            }
        }

        saveField2D(output_vector_dbl[i].name, vector_index,
                    vector_size, *output_vector_dbl[i].data);

    }

};
