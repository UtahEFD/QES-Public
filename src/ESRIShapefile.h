#pragma once

#include <cassert>
#include <vector>
#include "gdal.h"
#include "ogrsf_frmts.h"

class ESRIShapefile
{
public:
    ESRIShapefile();
    ESRIShapefile(const std::string &filename, const std::string &layerName);
    ~ESRIShapefile();

    void getLocalDomain( std::vector<float> &dim ) 
    {
        assert(dim.size() == 2);
        dim[0] = (int)ceil(maxBound[0] - minBound[0]);
        dim[1] = (int)ceil(maxBound[1] - minBound[1]);
    }
    
private:

    void loadVectorData();

    std::string m_filename;
    std::string m_layerName;
    
    GDALDataset *m_poDS;

    std::vector<float> minBound, maxBound;
};
