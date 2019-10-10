#include <fstream>
#include <cmath>

#include <iostream>

#include <netcdf>
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

#include <ogr_spatialref.h>
#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()

int main(int argc, char *argv[])
{
    std::string filename = argv[1];
    
    NcFile wrfInputFile( filename, NcFile::read );
        
    std::multimap<std::string,NcGroupAtt> globalAttributes = wrfInputFile.getAtts();
    
    // Grab the Stored end of the dimension and subtract 1.
    // xDim+1 is a pointer reference to the 2nd array value.
    // Same happens for yDim below.
    int xDim[2] = {1, 0}, yDim[2] = {1, 0};

    auto gblAttIter = globalAttributes.find("WEST-EAST_GRID_DIMENSION");
    gblAttIter->second.getValues( xDim+1 );
    xDim[1] -= 1;

    gblAttIter = globalAttributes.find("SOUTH-NORTH_GRID_DIMENSION");
    gblAttIter->second.getValues( yDim+1 );
    yDim[1] -= 1;
    
    // Compute nx and ny
    int nx = xDim[1] - xDim[0] + 1;
    int ny = yDim[1] - yDim[0] + 1;

    std::cout << "Domain is " << nx << " X " << ny << " cells." << std::endl;
    
    // Pull DX and DY
    double cellSize[2] = {1, 1};
    gblAttIter = globalAttributes.find("DX");
    gblAttIter->second.getValues( cellSize );
    
    gblAttIter = globalAttributes.find("DY");
    gblAttIter->second.getValues( cellSize+1 );
    
    std::cout << "Atmos Resolution (dx,dy) is ("<< cellSize[0] << ", " << cellSize[1] << ")" << std::endl;

    // 
    // Fire Mesh Terrain Nodes
    //
    int fm_nt = wrfInputFile.getVar("FXLONG").getDim(0).getSize();
    int fm_ny = wrfInputFile.getVar("FXLONG").getDim(1).getSize();
    int fm_nx = wrfInputFile.getVar("FXLONG").getDim(2).getSize();

    std::vector<size_t> startIdx = {0,0,0,0};
    std::vector<size_t> counts = {1,
                                  static_cast<unsigned long>(fm_ny),
                                  static_cast<unsigned long>(fm_nx)};

    std::vector<double> fxlong( fm_nx * fm_ny );
    std::vector<double> fxlat( fm_nx * fm_ny );
    wrfInputFile.getVar("FXLONG").getVar(startIdx, counts, fxlong.data());
    wrfInputFile.getVar("FXLAT").getVar(startIdx, counts, fxlat.data());
    
    std::vector<double> fmHeight( fm_nx * fm_ny );
    wrfInputFile.getVar("ZSF").getVar(startIdx, counts, fmHeight.data());

    int sizeHGT_x = wrfInputFile.getVar("HGT").getDim(2).getSize();
    int sizeHGT_y = wrfInputFile.getVar("HGT").getDim(1).getSize();
    int sizeZSF_x = wrfInputFile.getVar("ZSF").getDim(2).getSize();
    int sizeZSF_y = wrfInputFile.getVar("ZSF").getDim(1).getSize();

    // std::cout << "sizes: " << sizeHGT_x << ", " << sizeHGT_y << ", " << sizeZSF_x << ", " << sizeZSF_y << std::endl;

    float sr_x = sizeZSF_x/(sizeHGT_x+1);
    float sr_y = sizeZSF_y/(sizeHGT_y+1);

    // std::cout << "sr_x, sr_y = (" << sr_x << ", " << sr_y << ")" << std::endl;

    float dxf = cellSize[0] / sr_x;
    float dyf = cellSize[1] / sr_y;    
    // Then dxf=DX/sr_x, dyf=DY/sr_y

    std::cout << "Fire Mesh (dxf, dyf) = (" << dxf << ", " << dyf << ")" << std::endl;
    
    double minHt = std::numeric_limits<double>::max(),
        maxHt = std::numeric_limits<double>::min();
    
    for (int i=0; i<fm_nx; i++) {
        for (int j=0; j<fm_ny; j++) {
            
            int l_idx = i + j*fm_nx;

            if (fmHeight[l_idx] > maxHt) maxHt = fmHeight[l_idx];
            if (fmHeight[l_idx] < minHt) minHt = fmHeight[l_idx];
        }
    }

    double rangeHt = maxHt - minHt;
    std::cout << "Min Ht: " << minHt << ", Max Ht: " << maxHt << std::endl;

    int UTMZone = (int)floor((fxlong[0] + 180) / 6) + 1;
    
    std::cout << "UTM Zone: " << UTMZone << std::endl;
    std::cout << "(Lat,Long) at [0][0] = " << fxlat[0] << ", " << fxlong[0] << std::endl;   // 524972.33, 3376924.26
    std::cout << "(Lat,Long) at [nx-1][0] = " << fxlat[fm_nx-1] << ", " << fxlong[nx-1] << std::endl;
    std::cout << "(Lat,Long) at [0][ny-1] = " << fxlat[(fm_ny-1)*fm_nx] << ", " << fxlong[(fm_ny-1)*fm_nx] << std::endl;
    std::cout << "(Lat,Long) at [nx-1][ny-1] = " << fxlat[fm_nx-1 + (fm_ny-1)*fm_nx] << ", " << fxlong[fm_nx-1 + (fm_ny-1)*fm_nx] << std::endl;

    // [0] is lower left, [1] is pixel width, [2] is , [3] is 
    // Need to automate this conversion
    double adfGeoTransform[6] = {  524972.33, dxf, 0, 3376924.26, 0, -dyf };

    // is it possible to create a GDAL DS from this info?
    // write out GDAL file of fire mesh here as first pass.

   GDALAllRegister(); 

    const char *pszFormat = "GTiff";
    GDALDriver *poDriver;
    char **papszMetadata;
    poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
    if( poDriver == NULL ) {
        std::cerr << "No GTiff driver." << std::endl;
        exit( 1 );
    }
    
    papszMetadata = poDriver->GetMetadata();

    GDALDataset *poDstDS;
    char **papszOptions = NULL;

    poDstDS = poDriver->Create( "WRFOut.tiff", fm_nx, fm_ny, 1, GDT_Byte, papszOptions );

    OGRSpatialReference oSRS;
    char *pszSRS_WKT = NULL;
    GDALRasterBand *poBand;

    std::vector<GByte> abyRaster( fm_nx * fm_ny );

    for (int i=0; i<fm_nx; i++) {
        for (int j=0; j<fm_ny; j++) {
            
            int l_idx = i + j*fm_nx;
            abyRaster[ l_idx ] = (GByte)( (int)floor( ((fmHeight[l_idx]-minHt)/rangeHt) * 255 ) );
        }
    }
    std::cout << "Done." << std::endl;
    
    poDstDS->SetGeoTransform( adfGeoTransform );

//  NC_GLOBAL#TRUELAT1=30
//  NC_GLOBAL#TRUELAT2=34
//  NC_GLOBAL#CEN_LAT=30.533249
//  NC_GLOBAL#MOAD_CEN_LAT=30.53326
//  NC_GLOBAL#CEN_LON=-86.730408
      // oSRS.SetLCC(double dfStdP1, double dfStdP2, double dfCenterLat, double dfCenterLong, double dfFalseEasting, double dfFalseNorthing)

    oSRS.SetLCC(30.0, 34.0, 30.533249, -86.730408, 0.0, 0.0);
    
    oSRS.SetUTM( UTMZone, TRUE );
    oSRS.SetWellKnownGeogCS( "WGS84" );
    oSRS.exportToWkt( &pszSRS_WKT );
    poDstDS->SetProjection( pszSRS_WKT );
    
    CPLFree( pszSRS_WKT );

    poBand = poDstDS->GetRasterBand(1);
    poBand->RasterIO( GF_Write, 0, 0, fm_nx, fm_ny,
                      abyRaster.data(), fm_nx, fm_ny, GDT_Byte, 0, 0 );
    
    /* Once we're done, close properly the dataset */
    GDALClose( (GDALDatasetH) poDstDS );
}

