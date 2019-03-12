#include <iostream>
#include "ESRIShapefile.h"

int main(int argc, char *argv[])
{
    // take first arg as shp file to load
    ESRIShapefile shp( argv[1], argv[2] );
 
    exit(EXIT_SUCCESS);
}

