#pragma once

#include <vector>
#include "gdal.h"
#include "ogrsf_frmts.h"

class ESRIShapefile
{
public:
    ESRIShapefile();
    ESRIShapefile(const std::string &filename);
    ~ESRIShapefile();
    
private:

    void loadVectorData();

    std::string m_filename;
    GDALDataset *m_poDS;
};
