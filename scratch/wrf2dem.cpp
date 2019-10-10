#include <iostream>
#include <fstream>
#include <cmath>

#include <boost/filesystem.hpp>
namespace filesys = boost::filesystem;

#include <netcdf>
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

#include <ogr_spatialref.h>
#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cerr << "Missing required arguments:\n\twrf2dem WRF_inputfilename DEM_outputfilename" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::string filename = argv[1];
    std::string outFilename = argv[2];
    filesys::path outFilenamePath(outFilename);

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

    std::cout << "WRF Atmos Domain is " << nx << " X " << ny << " cells." << std::endl;
    
    // Pull DX and DY
    double cellSize[2] = {1, 1};
    gblAttIter = globalAttributes.find("DX");
    gblAttIter->second.getValues( cellSize );
    
    gblAttIter = globalAttributes.find("DY");
    gblAttIter->second.getValues( cellSize+1 );
    
    std::cout << "WRF Atmos Resolution (dx,dy) is ("<< cellSize[0] << ", " << cellSize[1] << ")" << std::endl;

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

    std::cout << "WRF Fire Mesh Domain is " << fm_nx << " X " << fm_ny << std::endl;
    std::cout << "WRF Fire Mesh Resolution (dx, dy) is (" << dxf << ", " << dyf << ")" << std::endl;
    
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
    std::cout << "Terrain Min Ht: " << minHt << ", Max Ht: " << maxHt << std::endl;

    int UTMZone = (int)floor((fxlong[0] + 180) / 6) + 1;
    
    std::cout << "UTM Zone: " << UTMZone << std::endl;
    std::cout << "(Lat,Long) at [0][0] = " << fxlat[0] << ", " << fxlong[0] << std::endl;   // 524972.33, 3376924.26
    std::cout << "(Lat,Long) at [nx-1][0] = " << fxlat[fm_nx-1] << ", " << fxlong[nx-1] << std::endl;
    std::cout << "(Lat,Long) at [0][ny-1] = " << fxlat[(fm_ny-1)*fm_nx] << ", " << fxlong[(fm_ny-1)*fm_nx] << std::endl;
    std::cout << "(Lat,Long) at [nx-1][ny-1] = " << fxlat[fm_nx-1 + (fm_ny-1)*fm_nx] << ", " << fxlong[fm_nx-1 + (fm_ny-1)*fm_nx] << std::endl;

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

    poDstDS = poDriver->Create( outFilename.c_str(), fm_nx, fm_ny, 1, GDT_Byte, papszOptions );

    OGRSpatialReference oSRS;
    GDALRasterBand *poBand;

    std::vector<GByte> abyRaster( fm_nx * fm_ny );

    for (int i=0; i<fm_nx; i++) {
        for (int j=0; j<fm_ny; j++) {
            
            int l_idx = i + j*fm_nx;
            abyRaster[ l_idx ] = (GByte)( (int)floor( ((fmHeight[l_idx]-minHt)/rangeHt) * 255 ) );
        }
    }
    std::cout << "Done." << std::endl;
    
    //  NC_GLOBAL#TRUELAT1=30
    //  NC_GLOBAL#TRUELAT2=34
    //  NC_GLOBAL#CEN_LAT=30.533249
    //  NC_GLOBAL#MOAD_CEN_LAT=30.53326
    //  NC_GLOBAL#CEN_LON=-86.730408
    // oSRS.SetLCC(double dfStdP1, double dfStdP2, double dfCenterLat, double dfCenterLong, double dfFalseEasting, double dfFalseNorthing)
    oSRS.SetLCC(30.0, 34.0, 30.533249, -86.730408, 0.0, 0.0);
    oSRS.SetWellKnownGeogCS( "WGS84" );

    char *exportResult;
    oSRS.exportToPrettyWkt(&exportResult);
    // std::cout << "oSRS: " << exportResult << std::endl;
    CPLFree(exportResult);
    
    // wgs84 coordinate system
    OGRSpatialReference wgs84sr;
    wgs84sr.SetWellKnownGeogCS( "WGS84" );
    wgs84sr.SetUTM( UTMZone, TRUE );
    
    wgs84sr.exportToPrettyWkt(&exportResult);
    // std::cout << "wgs84sr: " << exportResult << std::endl;
    CPLFree(exportResult);

    // set the transform wgs84_to_utm and do the transform
    //transform_WGS84_To_UTM = osr.CoordinateTransformation(wgs84_cs,utm_cs)
    OGRCoordinateTransformation *ogrCoordXform = OGRCreateCoordinateTransformation(&oSRS, &wgs84sr);
        
    double lats2eastings[1] = { 30.5244 };
    double longs2northings[1] = { -86.7397 };
    ogrCoordXform->Transform(1, lats2eastings, longs2northings);
    
    // std::cout << "UTM: " << lats2eastings[0] << ", " << longs2northings[0] << std::endl;

    // [0] is lower left, [1] is pixel width, [2] is , [3] is 
    // Need to automate this conversion
    double adfGeoTransform[6] = { lats2eastings[0], dxf, 0, longs2northings[0], 0, -dyf };
    poDstDS->SetGeoTransform( adfGeoTransform );

    char *pszSRS_WKT = NULL;
    wgs84sr.exportToWkt( &pszSRS_WKT );
    poDstDS->SetProjection( pszSRS_WKT );
    CPLFree( pszSRS_WKT );

    poBand = poDstDS->GetRasterBand(1);
    poBand->RasterIO( GF_Write, 0, 0, fm_nx, fm_ny,
                      abyRaster.data(), fm_nx, fm_ny, GDT_Byte, 0, 0 );
    
    /* Once we're done, close properly the dataset */
    GDALClose( (GDALDatasetH) poDstDS );

    std::string xmlOutput = outFilenamePath.filename().string() + ".xml";
    std::cout << "\twriting XML data to " << xmlOutput << std::endl;

    std::string stage1XML = "<simulationParameters>\n\t<DEM>";
    std::string stage2XML = "</DEM>\n\t<halo_x> 40.0 </halo_x>\n\t<halo_y> 40.0 </halo_y>\n";
    std::string stage3XML = "\t<verticalStretching> 0 </verticalStretching>\n\t<totalTimeIncrements> 1 </totalTimeIncrements>\n\t<UTCConversion> 0 </UTCConversion>\n\t<Epoch> 1510930800 </Epoch>\n\t<rooftopFlag> 0 </rooftopFlag>\n\t<upwindCavityFlag> 0 </upwindCavityFlag>\n\t<streetCanyonFlag> 0 </streetCanyonFlag>\n\t<streetIntersectionFlag> 0 </streetIntersectionFlag>\n\t<wakeFlag> 0 </wakeFlag>\n\t<sidewallFlag> 0 </sidewallFlag>\n\t<maxIterations> 500 </maxIterations>\n\t<residualReduction> 3 </residualReduction>\n\t<meshTypeFlag> 0 </meshTypeFlag> <!-- cut cell -->\n\t<useDiffusion> 0 </useDiffusion>\n\t<domainRotation> 0 </domainRotation>\n\t<UTMX> 0 </UTMX>\n\t<UTMY> 0 </UTMY>\n\t<UTMZone> 1 </UTMZone>\n\t<UTMZoneLetter> 17 </UTMZoneLetter>\n\t</simulationParameters>\n<metParams>\n\t<metInputFlag> 0 </metInputFlag>\n\t<num_sites> 1 </num_sites>\n\t<maxSizeDataPoints> 1 </maxSizeDataPoints>\n\t<siteName> sensor1 </siteName>\n\t<fileName> sensor1.inp </fileName>\n\t<z0_domain_flag> 0 </z0_domain_flag>            <!-- Distribution of surface roughness for domain (0-uniform, 1-custom -->\n\t<sensor>\n\t<site_coord_flag> 1 </site_coord_flag> 				<!-- Sensor site coordinate system (1=QUIC, 2=UTM, 3=Lat/Lon) -->\n\t<site_xcoord> 1.0  </site_xcoord> 					<!-- x component of site location in QUIC domain (m) (if site_coord_flag = 1) -->\n\t<site_ycoord> 1.0 </site_ycoord> 				<!-- y component of site location in QUIC domain (m) (if site_coord_flag = 1)-->\n\t<site_UTM_x> 2.0 </site_UTM_x> 								<!-- x components of site coordinate in UTM (if site_coord_flag = 2) -->\n\t<site_UTM_y> 2.0 </site_UTM_y> 								<!-- y components of site coordinate in UTM (if site_coord_flag = 2)-->\n\t<site_UTM_zone> 0 </site_UTM_zone> 						<!-- UTM zone of the sensor site (if site_coord_flag = 2)-->\n\t <boundaryLayerFlag> 1 </boundaryLayerFlag> 			<!-- Site boundary layer flag (1-log, 2-exp, 3-urban canopy, 4-data entry) -->\n\t<siteZ0> 0.1 </siteZ0> 									<!-- Site z0 -->\n\t<reciprocal> 0.0 </reciprocal> 						<!-- Reciprocal Monin-Obukhov Length (1/m) -->\n\t<height> 10.0 </height> 										<!-- Height of the sensor -->\n\t<speed> 5.0 </speed> 											<!-- Measured speed at the sensor height -->\n\t<direction> 360.0 </direction> 						<!-- Wind direction of sensor -->\n\t</sensor>                       	<!-- Wnd of sensor section -->\n</metParams>\n<fileOptions>\n\t<outputFlag>1</outputFlag>\n\t<outputFields>u</outputFields> \n\t<outputFields>v</outputFields> \n\t<outputFields>w</outputFields>\n\t<outputFields>icell</outputFields>   \n\t<massConservedFlag> 0 </massConservedFlag>\n\t<sensorVelocityFlag> 0 </sensorVelocityFlag>\n\t<staggerdVelocityFlag> 0 </staggerdVelocityFlag>\n</fileOptions>\n";
    
    ofstream xmlout;
    xmlout.open( xmlOutput, std::ofstream::out );

    xmlout << stage1XML;
    xmlout << outFilenamePath.filename();
    xmlout << stage2XML;
    xmlout << "\t<domain> " << fm_nx << " " << fm_ny << " " << 300 << "</domain>\n\t<cellSize> " << dxf << " " << dyf << " 1.0 </cellSize>\n";
    xmlout << stage3XML;
    xmlout.close();
}

