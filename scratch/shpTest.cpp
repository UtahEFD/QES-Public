#include <iostream>
#include "ESRIShapefile.h"

int main(int argc, char *argv[])
{
    std::vector< std::vector <polyVert> > shpPolygons;

    // take first arg as shp file to load
    ESRIShapefile shp( argv[1], argv[2], shpPolygons );
 
    exit(EXIT_SUCCESS);
}

