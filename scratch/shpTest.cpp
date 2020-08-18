#include <iostream>
#include "ESRIShapefile.h"

int main(int argc, char *argv[])
{
    std::vector< std::vector <polyVert> > shpPolygons;
    std::vector <float> building_height;
    float heightFactor = 1.0;

    // take first arg as shp file to load
    ESRIShapefile shp( argv[1], argv[2], shpPolygons, building_height, heightFactor );

    exit(EXIT_SUCCESS);
}
