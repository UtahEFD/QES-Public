#pragma once

#include <cassert>
#include <vector>
#include "gdal.h"
#include "ogrsf_frmts.h"
#include <limits>

#include "PolygonVertex.h"

class ESRIShapefile
{
public:
    ESRIShapefile();
    ESRIShapefile(const std::string &filename, const std::string &layerName,
                  std::vector< std::vector< polyVert > >& polygons, std::vector <float> &building_height, float heightFactor);
    ~ESRIShapefile();

    void getLocalDomain( std::vector<float> &dim )
    {
        assert(dim.size() == 2);
        dim[0] = (int)ceil(maxBound[0] - minBound[0]);
        dim[1] = (int)ceil(maxBound[1] - minBound[1]);
    }

    void getMinExtent( std::vector<float> &ext )
    {
        assert(ext.size() == 2);
        ext[0] = minBound[0];
        ext[1] = minBound[1];
    }

private:

    void loadVectorData( std::vector< std::vector< polyVert > > &polygons, std::vector <float> &building_height, float heightFactor );

    std::string m_filename;
    std::string m_layerName;

    GDALDataset *m_poDS;

    std::vector<float> minBound, maxBound;
};
