#include <iostream>
#include <fstream>
#include <sstream>

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

class profData 
{
public:
    float zCoord;
    float ws, wd;
};

class stationData 
{
public:
    stationData() {}
    ~stationData() {}

    float xCoord, yCoord;
    std::vector< std::vector< profData > > profiles;

private:
};

const double mPI = 3.14159;

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

    // Atmospheric mesh size is stored in GRID_DIMENSIONs
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
    GDALRasterBand *poBand;
    char **papszOptions = NULL;
    poDstDS = poDriver->Create( outFilename.c_str(), fm_nx, fm_ny, 1, GDT_UInt16, papszOptions );

    std::vector<float> abyRaster( fm_nx * fm_ny );

    for (int i=0; i<fm_nx; i++) {
        for (int j=0; j<fm_ny; j++) {
            
            int l_idx = i + j*fm_nx;
            abyRaster[ l_idx ] = fmHeight[l_idx];
        }
    }
    std::cout << "Done." << std::endl;
    
    // 
    // Setting up WRF-based SpatialReference
    //
    //    lat1 = d.TRUELAT1
    //    lat2 = d.TRUELAT2
    //    lat0 = d.MOAD_CEN_LAT
    //    lon0 = d.STAND_LON
    //    clat = d.CEN_LAT
    //    clon = d.CEN_LON
    //    csr = osr.SpatialReference()
    //    proj4 = '+proj=lcc +lat_1=%.10f +lat_2=%.10f +lat_0=%.10f +lon_0=%.10f +a=6370000.0 +b=6370000.0' % (lat1,lat2,lat0 \
,lon0)
    //    logging.info('proj4: %s' % proj4)
    //    csr.ImportFromProj4(proj4)
    //    ll_proj = pyproj.Proj('+proj=latlong +datum=WGS84')
//     wrf_proj = pyproj.Proj(proj4)

    double lat1, lat2, lat0, lon0, clat, clon;
    gblAttIter = globalAttributes.find("TRUELAT1");
    gblAttIter->second.getValues( &lat1 );
    
    gblAttIter = globalAttributes.find("TRUELAT2");
    gblAttIter->second.getValues( &lat2 );

    gblAttIter = globalAttributes.find("MOAD_CEN_LAT");
    gblAttIter->second.getValues( &lat0 );
    
    gblAttIter = globalAttributes.find("STAND_LON");
    gblAttIter->second.getValues( &lon0 );
    
    gblAttIter = globalAttributes.find("CEN_LAT");
    gblAttIter->second.getValues( &clat );
    
    gblAttIter = globalAttributes.find("CEN_LON");
    gblAttIter->second.getValues( &clon );

    // WRF coordinates are unique projections.  Use Lambert Conformal
    // Conic projections.
    std::ostringstream proj4ss;
    proj4ss << "+proj=lcc +lat_1=" << lat1 << " +lat_2=" << lat2 << " +lat_0=" << lat0 << " +lon_0=" << lon0 << " +a=6370000.0 +b=6370000.0";

    std::cout << "Initializing WRF Spatial Reference from PROJ4 string: " << proj4ss.str() << std::endl;
    OGRSpatialReference wrfSpatialRef;
    wrfSpatialRef.importFromProj4( proj4ss.str().c_str() );

    char *exportResult;
    wrfSpatialRef.exportToPrettyWkt(&exportResult);
    std::cout << "WRF Spatial Reference: " << exportResult << std::endl;
    CPLFree(exportResult);

    std::string proj4_spherLatLon = "+proj=latlong +a=6370000 +b=6370000";
    OGRSpatialReference sr_SpherLatLon;
    sr_SpherLatLon.importFromProj4( proj4_spherLatLon.c_str() );

 //    csr.ImportFromProj4(proj4)
 //    ll_proj = pyproj.Proj('+proj=latlong +datum=WGS84')
 //    wrf_proj = pyproj.Proj(proj4)

    OGRSpatialReference wgs84;
    std::ostringstream outString;
    outString << "UTM " << UTMZone << " (WGS84) in northern hemisphere.";
    wgs84.SetProjCS( outString.str().c_str() );
    wgs84.SetWellKnownGeogCS( "WGS84" );
    wgs84.SetUTM( UTMZone, TRUE );
    // wgs84.importFromProj4( "+proj=latlong +datum=WGS84" );
     
    OGRSpatialReference latLongProj;
    std::string projString = "+proj=latlong +datum=WGS84";
    latLongProj.importFromProj4( projString.c_str() );
    

    // # geotransform
    // e,n = pyproj.transform(ll_proj,wrf_proj,clon,clat)
    // dx_atm = d.DX
    // dy_atm = d.DY
    // nx_atm = d.dimensions['west_east'].size
    // ny_atm = d.dimensions['south_north'].size
    // x0_atm = -nx_atm / 2. * dx_atm + e
    // y1_atm = ny_atm / 2. * dy_atm + n
    // geotransform_atm = (x0_atm,dx_atm,0,y1_atm,0,-dy_atm)

//    OGRCoordinateTransformation *wrfCoordXform = OGRCreateCoordinateTransformation(&ll_proj, &wrfSpatialRef);    
//    std::cout << "clon, clat = " << clon << ", " << clat << std::endl;
//    wrfCoordXform->Transform(1, &clon, &clat);
//    std::cout << "Transformed: clon, clat = " << clon << ", " << clat << std::endl;

    //  NC_GLOBAL#TRUELAT1=30
    //  NC_GLOBAL#TRUELAT2=34
    //  NC_GLOBAL#CEN_LAT=30.533249
    //  NC_GLOBAL#MOAD_CEN_LAT=30.53326
    //  NC_GLOBAL#CEN_LON=-86.730408
    // oSRS.SetLCC(double dfStdP1, double dfStdP2, double dfCenterLat, double dfCenterLong, double dfFalseEasting, double dfFalseNorthing)

    // oSRS.SetLCC(30.0, 34.0, 30.533249, -86.730408, 0.0, 0.0);
    // oSRS.SetWellKnownGeogCS( "WGS84" );
    
    
    // wgs84 coordinate system
//    OGRSpatialReference wgs84sr;
//    wgs84sr.SetWellKnownGeogCS( "WGS84" );
//    wgs84sr.SetUTM( UTMZone, TRUE );
    
    // set the transform wgs84_to_utm and do the transform
    //transform_WGS84_To_UTM =
    //osr.CoordinateTransformation(wgs84_cs,utm_cs)
    // src to dst
    OGRCoordinateTransformation *ogrCoordXform1of2 = OGRCreateCoordinateTransformation(&wrfSpatialRef, &sr_SpherLatLon);
    OGRCoordinateTransformation *ogrCoordXform2of2 = OGRCreateCoordinateTransformation(&sr_SpherLatLon, &wgs84);

    OGRCoordinateTransformation *ogrCoordXform3 = OGRCreateCoordinateTransformation(&wgs84, &wrfSpatialRef);
    // OGRCoordinateTransformation *ogrCoordXform3 = OGRCreateCoordinateTransformation(&wrfSpatialRef, &wgs84);

    // OGRCoordinateTransformation *ogrCoordXform4 = OGRCreateCoordinateTransformation(&latLongProj, &wrfSpatialRef);
    OGRCoordinateTransformation *ogrCoordXform4 = OGRCreateCoordinateTransformation(&wrfSpatialRef, &latLongProj);
        
    // From: https://gdal.org/tutorials/osr_api_tut.html
    // Starting with GDAL 3.0, the axis order mandated by the
    // authority defining a CRS is by default honoured by the
    // OGRCoordinateTransformation class, and always exported in
    // WKT1. Consequently CRS created with the “EPSG:4326” or “WGS84”
    // strings use the latitude first, longitude second axis order.

    double lon2eastings[1] = { clon };
    double lat2northings[1] = { clat };

//    ogrCoordXform1of2->Transform(1, lon2eastings, lat2northings);
//    ogrCoordXform2of2->Transform(1, lon2eastings, lat2northings);

//    ogrCoordXform3->Transform(1, lon2eastings, lat2northings);
    ogrCoordXform4->Transform(1, lon2eastings, lat2northings);

    std::cout << "UTM: " << lon2eastings[0] << ", " << lat2northings[0] << std::endl;    

    nx = fm_nx;
    ny = fm_ny;
    
    int nx_atm = xDim[1];
    int ny_atm = yDim[1];    

    int srx = int(nx/(nx_atm+1));
    int sry = int(ny/(ny_atm+1));
    
    int nx_fire = nx - srx;
    int ny_fire = ny - sry;
    
    float dx_atm = cellSize[0];
    float dy_atm = cellSize[1];

    float dx_fire = dx_atm/(float)srx;
    float dy_fire = dy_atm/(float)sry;
    
    double t_x0_fire = -nx_fire / 2. * dx_fire + lon2eastings[0];
    double t_y1_fire = (ny_fire / 2. + sry) * dy_fire + lat2northings[0];

    std::cout << "Xform: " << srx << ", " << sry
              << "; " << nx_fire << ", " << ny_fire
              << "; " << dx_fire << ", " << dy_fire
              << "; " << t_x0_fire << ", " << t_y1_fire << std::endl;
    
    

    // nx_atm / 2. * dx_atm + e
    double x0_fire = t_x0_fire; // -fm_nx / 2.0 * dxf + lon2eastings[0]; 
    // ny_atm / 2. * dy_atm + n 
    double y1_fire = t_y1_fire; // -fm_ny / 2.0 * dyf + lat2northings[0];

    double geoTransform_fireMesh[6] = { x0_fire, dxf, 0, y1_fire, 0, -dyf };    
    poDstDS->SetGeoTransform( geoTransform_fireMesh );

    // [0] is lower left, [1] is pixel width, [2] is , [3] is 
    // Need to automate this conversion
//    double adfGeoTransform[6] = { lon2eastings[0], dxf, 0, lat2northings[0], 0, -dyf };
//    poDstDS->SetGeoTransform( adfGeoTransform );


    char *pszSRS_WKT = NULL;
    wgs84.exportToWkt( &pszSRS_WKT );
    poDstDS->SetProjection( pszSRS_WKT );
    CPLFree( pszSRS_WKT );

    poBand = poDstDS->GetRasterBand(1);
    poBand->RasterIO( GF_Write, 0, 0, fm_nx, fm_ny,
                      abyRaster.data(), fm_nx, fm_ny, GDT_UInt16, 0, 0 );
    
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


    //
    // remainder of code is for pulling out wind profiles
    //

    // Need atm mesh sizes, stored in nx_atm and ny_atm
    // top grid dimension stored in BOTTOM-TOP_GRID_DIMENSION
    int nz_atm = 1;
    gblAttIter = globalAttributes.find("BOTTOM-TOP_GRID_DIMENSION");
    gblAttIter->second.getValues( &nz_atm );
    
    std::cout << "Atmos NZ = " << nz_atm << std::endl;
    std::cout << "Atmos NY = " << ny_atm << std::endl;
    std::cout << "Atmos NX = " << nx_atm << std::endl;

    // Wind data vertical positions, stored in PHB and PH
    // 
    // For example, 360 time series, 41 vertical, 114 x 114 
    // SUBDATASET_12_DESC=[360x41x114x114] PH (32-bit floating-point)
    // SUBDATASET_13_NAME=NETCDF:"/scratch/Downloads/RXCwrfout_d07_2012-11-11_15-21":PHB
    // 
    // SUBDATASET_13_DESC=[360x41x114x114] PHB (32-bit floating-point)
    // SUBDATASET_14_NAME=NETCDF:"/scratch/Downloads/RXCwrfout_d07_2012-11-11_15-21":T
    //
    std::vector<size_t> atmStartIdx = {0,0,0,0};
    std::vector<size_t> atmCounts = {2,
                                     static_cast<unsigned long>(nz_atm),
                                     static_cast<unsigned long>(ny_atm),
                                     static_cast<unsigned long>(nx_atm)};

    std::vector<double> phbData( 2 * nz_atm * ny_atm * nx_atm );
    wrfInputFile.getVar("PHB").getVar(atmStartIdx, atmCounts, phbData.data());

    std::vector<double> phData( 2 * nz_atm * ny_atm * nx_atm );
    wrfInputFile.getVar("PH").getVar(atmStartIdx, atmCounts, phData.data());

    std::cout << "PHB and PH read in..." << std::endl;

    //
    // Calculate the height  (PHB + PH) / 9.81
    //
    std::vector<double> heightData( 2 * nz_atm * ny_atm * nx_atm );
    for (int t=0; t<2; t++) {
        for (int k=0; k<nz_atm; k++) { 
            for (int j=0; j<ny_atm; j++) {
                for (int i=0; i<nx_atm; i++) {
                    int l_idx = t*(nz_atm*ny_atm*nx_atm) + k*(ny_atm*nx_atm) + j*nx_atm + i;
                    heightData[ l_idx ] = (phbData[l_idx] + phData[l_idx]) / 9.81;
                }
            }
        }
    }
    std::cout << "Height computed." << std::endl;

    // use winds straight from atmospheric 

    // calculate number of altitudes to store, while reset
    atmCounts[1] = nz_atm - 1;

    //
    // Wind components are on staggered grid in U and V
    //
    // When these are read in, we will read one additional cell in X
    // and Y, for U and V, respectively.
    atmCounts[3] = nx_atm + 1;
    std::vector<double> UStaggered( 2 * (nz_atm-1) * ny_atm * (nx_atm+1) );
    wrfInputFile.getVar("U").getVar(atmStartIdx, atmCounts, UStaggered.data());
    
    atmCounts[2] = ny_atm + 1;
    atmCounts[3] = nx_atm;
    std::vector<double> VStaggered( 2 * (nz_atm-1) * (ny_atm+1) * nx_atm );
    wrfInputFile.getVar("V").getVar(atmStartIdx, atmCounts, VStaggered.data());
    atmCounts[2] = ny_atm;  // reset to ny_atm
    
    // But then, U and V are on standard nx and ny atmos mesh
    std::vector<double> U( 2 * (nz_atm-1) * ny_atm * nx_atm );
    std::vector<double> V( 2 * (nz_atm-1) * ny_atm * nx_atm );
    for (int t=0; t<2; t++) {
        for (int k=0; k<(nz_atm-1); k++) { 
            for (int j=0; j<ny_atm; j++) {
                for (int i=0; i<nx_atm; i++) {
                    
                    int l_idx = t*((nz_atm-1)*ny_atm*nx_atm) + k*(ny_atm*nx_atm) + j*nx_atm + i;

                    int l_xp1_idx = t*((nz_atm-1)*ny_atm*nx_atm) + k*(ny_atm*nx_atm) + j*nx_atm + i+1;
                    int l_yp1_idx = t*((nz_atm-1)*ny_atm*nx_atm) + k*(ny_atm*nx_atm) + (j+1)*nx_atm + i;
                    
                    U[l_idx] = 0.5 * (UStaggered[l_idx] + UStaggered[l_xp1_idx]);
                    V[l_idx] = 0.5 * (VStaggered[l_idx] + VStaggered[l_yp1_idx]);
                }
            }
        }
    }
    

    std::cout << "Staggered wind input." << std::endl;

    // Calculate CoordZ
    std::vector<double> coordZ( 2 * (nz_atm-1) * ny_atm * nx_atm );
    for (int t=0; t<2; t++) {
        for (int k=0; k<(nz_atm-1); k++) { 
            for (int j=0; j<ny_atm; j++) {
                for (int i=0; i<nx_atm; i++) {
                    int l_idx = t*((nz_atm-1)*ny_atm*nx_atm) + k*(ny_atm*nx_atm) + j*nx_atm + i;
                    int l_kp1_idx = t*((nz_atm-1)*ny_atm*nx_atm) + (k+1)*(ny_atm*nx_atm) + j*nx_atm + i;
                    coordZ[l_idx] = 0.5*(heightData[l_idx] + heightData[l_kp1_idx]);
                }
            }
        }
    }
    


    //
    // wind speed sqrt(u*u + v*v);
    //
    std::vector<double> wsData( 2 * (nz_atm-1) * ny_atm * nx_atm );
    for (int t=0; t<2; t++) {
        for (int k=0; k<(nz_atm-1); k++) { 
            for (int j=0; j<ny_atm; j++) {
                for (int i=0; i<nx_atm; i++) {
                    int l_idx = t*((nz_atm-1)*ny_atm*nx_atm) + k*(ny_atm*nx_atm) + j*nx_atm + i;
                    wsData[l_idx] = sqrt( U[l_idx]*U[l_idx] + V[l_idx]*V[l_idx] );
                }
            }
        }
    }

    std::cout << "Wind speed computed." << std::endl;


    //
    // compute wind direction
    //
    std::vector<double> wdData( 2 * (nz_atm-1) * ny_atm * nx_atm );
    for (int t=0; t<2; t++) {
        for (int k=0; k<(nz_atm-1); k++) { 
            for (int j=0; j<ny_atm; j++) {
                for (int i=0; i<nx_atm; i++) {
                    int l_idx = t*((nz_atm-1)*ny_atm*nx_atm) + k*(ny_atm*nx_atm) + j*nx_atm + i;

                    if (U[l_idx] > 0.0)
                        wdData[l_idx] = 270.0 - (180.0/mPI) * atan(V[l_idx]/U[l_idx]);
                    else
                        wdData[l_idx] = 90.0 - (180.0/mPI) * atan(V[l_idx]/U[l_idx]);
                }
            }
        }
    }


    // TODO
    // read LU -- depends on if reading the restart or the output file
    // roughness length...

    
    

    // % Here we lower minimum altitude to 0 in order to save computational space.
    // % For the same reason, topography below 50 cm will not be
    // % considered.


    // QUESTIONS:
    //
    // What should be choice for min WRF altitudein meter?

    // What is choice for max wrf site altitude - XML or quick vertical domain size

    // Do we plan to use more than 2 time series? -- specify times steps, in XML

    // What is mapping between ZSF fire mesh and the U, V space?
                    
    float minWRFAlt = 22;
    float maxWRFAlt = 330;

    std::vector< stationData > statData;

    std::cout << "nx_atm: " << nx_atm << ", ny_atm = " << ny_atm << std::endl;

    // sampling strategy
    int stepSize = 12;

    for (int yIdx=0; yIdx<ny_atm; yIdx+=stepSize) {
        for (int xIdx=0; xIdx<nx_atm; xIdx+=stepSize) {

            // StatData.CoordX(Stat) = x;
            // StatData.CoordY(Stat) = y;
            stationData sd;
            sd.xCoord = xIdx;
            sd.yCoord = yIdx;
            sd.profiles.resize(2);  // 2 time series

            for (int t=0; t<2; t++) {
                
                // At this X, Y, look through all heights and
                // accumulate the heights that exist between our min
                // and max
                for (int k=0; k<(nz_atm-1); k++) {

                    int l_idx = t*((nz_atm-1)*ny_atm*nx_atm) + k*(ny_atm*nx_atm) + yIdx*nx_atm + xIdx;

                    if (coordZ[l_idx] >= minWRFAlt && coordZ[l_idx] <= maxWRFAlt) {
                        
                        profData profEl;
                        profEl.zCoord = coordZ[ l_idx ];
                        profEl.ws = wsData[ l_idx ];
                        profEl.wd = wdData[ l_idx ];

                        sd.profiles[t].push_back( profEl );
                    }
                }
            }
            
            statData.push_back( sd );
        }
    }

    std::cout << "Size of stat data: " << statData.size() << std::endl;
    for (int i=0; i<statData.size(); i++) {
        std::cout << "Station " << i << " (" << statData[i].xCoord << ", " << statData[i].yCoord << ")" << std::endl;
        for (int t=0; t<2; t++) {
            std::cout << "\tTime Series: " << t << std::endl;
            for (int p=0; p<statData[i].profiles[t].size(); p++) {
                std::cout << "\t" << statData[i].profiles[t][p].zCoord << ", " << statData[i].profiles[t][p].ws << ", " << statData[i].profiles[t][p].wd << std::endl;
            }
        }
    }

}

    
            
#if 0        
        StatData.CoordZ{Stat} = reshape(SimData.CoordZ(y,x,levelk_max,:),numel(levelk_max),size(SimData.CoordZ,4));
        StatData.nz(Stat) = size(StatData.CoordZ{Stat},1);
        
        StatData.WS{Stat} = reshape(SimData.WS(y,x,levelk_max,:),numel(levelk_max),size(SimData.WS,4));
        StatData.WD{Stat} = reshape(SimData.WD(y,x,levelk_max,:),numel(levelk_max),size(SimData.WD,4));   
#endif


