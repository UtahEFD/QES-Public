/** \file "WRFInput.cpp" input data header file. 
    \author Pete Willemsen, Matthieu 

    Copyright (C) 2019 Pete Willemsen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
*/

#include <fstream>
#include <cmath>

#include "WRFInput.h"

#include <ogr_spatialref.h>
#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()

WRFInput::WRFInput(const std::string& filename)
    : wrfInputFile( filename, NcFile::read ),
      m_minWRFAlt( 22 ), m_maxWRFAlt( 330 ), m_maxTerrainSize( 10001 ), m_maxNbStat( 156 ),
      m_TerrainFlag(1), m_BdFlag(0), m_VegFlag(0), m_Z0Flag(2)
{
    std::cout << "WRF Input Processor - reading data from " << filename << std::endl;

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
    fm_ny = wrfInputFile.getVar("FXLONG").getDim(1).getSize();
    fm_nx = wrfInputFile.getVar("FXLONG").getDim(2).getSize();

    std::vector<size_t> startIdx = {0,0,0,0};
    std::vector<size_t> counts = {1,
                                  static_cast<unsigned long>(fm_ny),
                                  static_cast<unsigned long>(fm_nx)};

    std::vector<double> fxlong( fm_nx * fm_ny );
    std::vector<double> fxlat( fm_nx * fm_ny );
    wrfInputFile.getVar("FXLONG").getVar(startIdx, counts, fxlong.data());
    wrfInputFile.getVar("FXLAT").getVar(startIdx, counts, fxlat.data());
    
    fmHeight.resize( fm_nx * fm_ny );
    wrfInputFile.getVar("ZSF").getVar(startIdx, counts, fmHeight.data());

    int sizeHGT_x = wrfInputFile.getVar("HGT").getDim(2).getSize();
    int sizeHGT_y = wrfInputFile.getVar("HGT").getDim(1).getSize();
    int sizeZSF_x = wrfInputFile.getVar("ZSF").getDim(2).getSize();
    int sizeZSF_y = wrfInputFile.getVar("ZSF").getDim(1).getSize();

    float sr_x = sizeZSF_x/(sizeHGT_x+1);
    float sr_y = sizeZSF_y/(sizeHGT_y+1);

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

#if 0

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
#endif
    
    std::vector<float> abyRaster( fm_nx * fm_ny );
    for (int i=0; i<fm_nx; i++) {
        for (int j=0; j<fm_ny; j++) {
            
            int l_idx = i + j*fm_nx;
            abyRaster[ l_idx ] = fmHeight[l_idx];
        }
    }
    std::cout << "Done." << std::endl;
    
#if 0
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
    //    proj4 = '+proj=lcc +lat_1=%.10f +lat_2=%.10f +lat_0=%.10f +lon_0=%.10f +a=6370000.0 +b=6370000.0' % (lat1,lat2,lat0,lon0)
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
#endif

    double clat, clon;
    gblAttIter = globalAttributes.find("CEN_LAT");
    gblAttIter->second.getValues( &clat );
    
    gblAttIter = globalAttributes.find("CEN_LON");
    gblAttIter->second.getValues( &clon );

    double lon2eastings[1] = { clon };
    double lat2northings[1] = { clat };


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

#if 0
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
#endif

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
                        wdData[l_idx] = 270.0 - (180.0/c_PI) * atan(V[l_idx]/U[l_idx]);
                    else
                        wdData[l_idx] = 90.0 - (180.0/c_PI) * atan(V[l_idx]/U[l_idx]);
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

#if 0
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
#endif

}

WRFInput::~WRFInput()
{
}

void WRFInput::readDomainInfo()
{
#if 0
    // Read domain dimensions, terrain elevation, wind data, land-use
    // and Z0 from WRF output.
    //
    // Also need to make sure we pull information from the Fire Mesh.
    //
    // NC_GLOBAL - WEST-EAST_GRID_DIMENSION,
    // SOUTH-NORTH_GRID_DIMENSION
    //
    //   NC_GLOBAL#CEN_LAT=30.533249
    //   NC_GLOBAL#CEN_LON=-86.730408
    //
    // From: https://www.openwfm.org/images/archive/e/ec/20101104161506%21Users_guide_chap-wrf-fire.pdf
    // ZSF - The variable ZSF contains high resolution terrain height
    // information similar to that in the HGT variable present in
    // atmospheric simulations;

    // FXLONG, FXLAT, ZSF ‐ coordinates of fire mesh nodes 
    // FUEL_TIME, BBB, BETAFL, PHIWC, R_0, FGIP, ISCHAP – fuel coefficients 

    // Need to convert the terrain elevation directly into a Mesh that
    // can be used by code.

    // FXLONG
    // FXLAT
    // FWH
    // FZ0
    

    // Possibility to crop domain borders by
    // adding a varargin (expected format: [Xstart,Xend;YStart,Yend]).
    // Layers of data are recorder in the following format: (row,col)
    // = (ny,nx).

    //SimData.Clock = ncread(WRFFile,'Times'); % Time in vector format
                                                
    // Read the WRF NetCDF file as read-only (default option in the
    // following call).
    //    try
    //    {
        // Open the file for read access

        // The netCDF file is automatically closed by the NcFile destructor
    // wrfInputFile( "/scratch/Downloads/RXCwrfout_d07_2012-11-11_15-21", NcFile::read );

    // Retrieve the variable named "Times": char Times(Time, DateStrLen) ;
    NcVar simDataClock = wrfInputFile.getVar("Times");
    if (simDataClock.isNull()) return;

    // external = infile->getDim(name);
    // 
    // input->getDimensionSize("x",grid.nx);
    // input->getDimensionSize("y",grid.ny);
    // 
    // input->getVariableData("u",start,count,wind.u);

    std::cout << "Number of attributes: " << wrfInputFile.getAttCount() << std::endl;
    std::multimap<std::string,NcGroupAtt> globalAttributes = wrfInputFile.getAtts();

    // You can see all of the attributes with the following snippet:
    // for (auto i=globalAttributes.cbegin(); i!=globalAttributes.cend(); i++) {
    // std::cout << "Attribute Name: " << i->first << std::endl;
    // }
        
    auto gblAttIter = globalAttributes.find("WEST-EAST_GRID_DIMENSION");
    xDim[0] = 1;

    // Grab the Stored end of the dimension and subtract 1.
    // xDim+1 is a pointer reference to the 2nd array value.
    // Same happens for yDim below.
    gblAttIter->second.getValues( xDim+1 );
    xDim[1] -= 1;

    gblAttIter = globalAttributes.find("SOUTH-NORTH_GRID_DIMENSION");
    yDim[0] = 1;
    gblAttIter->second.getValues( yDim+1 );
    yDim[1] -= 1;
    
    // Compute nx and ny
    nx = xDim[1] - xDim[0] + 1;
    ny = yDim[1] - yDim[0] + 1;

    std::cout << "Domain is " << nx << " X " << ny << " cells." << std::endl;
    
    // Pull out DX and DY
    double cellSize[2];
    gblAttIter = globalAttributes.find("DX");
    gblAttIter->second.getValues( cellSize );
    
    gblAttIter = globalAttributes.find("DY");
    gblAttIter->second.getValues( cellSize+1 );
    
    m_dx = cellSize[0];
    m_dy = cellSize[1];
    std::cout << "Resolution (dx,dy) is ("<< m_dx << ", " << m_dy << ")" << std::endl;

           
// % If new domain borders are defined
// if nargin == 3 
        
//    NewDomainCorners = varargin{1};
    
//    XSTART_New = NewDomainCorners(1,1); XEND_New = NewDomainCorners(1,2); 
//    YSTART_New = NewDomainCorners(2,1); YEND_New = NewDomainCorners(2,2);
    
//    SimData.nx = XEND_New - XSTART_New +1;
//    SimData.ny = YEND_New - YSTART_New +1; 
    
//    SimData.OLD_XSTART = SimData.XSTART; SimData.OLD_XEND = SimData.XEND;
//    SimData.OLD_YSTART = SimData.YSTART; SimData.OLD_YEND = SimData.YEND;
        
//    SimData.XSTART = XSTART_New; SimData.XEND = XEND_New;
//    SimData.YSTART = YSTART_New; SimData.YEND = YEND_New;
      
// end


// Relief = ncread(WRFFile,'HGT');
// SimData.Relief = Relief(SimData.XSTART:SimData.XEND,SimData.YSTART:SimData.YEND,1)'; 

    // 
    // Fire Mesh Terrain Nodes
    //

    // First, need the lat/long of the nodes

    int fm_nt = wrfInputFile.getVar("FXLONG").getDim(0).getSize();
    int fm_ny = wrfInputFile.getVar("FXLONG").getDim(1).getSize();
    int fm_nx = wrfInputFile.getVar("FXLONG").getDim(2).getSize();

    std::vector<size_t> startIdx = {0,0,0,0};
    std::vector<size_t> counts = {1,
                                  static_cast<unsigned long>(fm_ny),
                                  static_cast<unsigned long>(fm_nx)};

    //SUBDATASET_214_NAME=NETCDF:"RXCwrfout_d07_2012-11-11_15-21":FXLONG
    // SUBDATASET_214_DESC=[360x1150x1150] FXLONG (32-bit floating-point)
    // FXLONG
    // std::vector<NcDim> fxlongDims = wrfInputFile.getVar("FXLONG").getDims();
    // for (int i=0; i<fxlongdims.size(); i++) {
    //    std::cout << "Dim: " << fxlongdims[i].getName() << ", ";
    //     if (fxlongdims[i].isUnlimited())
    //      std::cout << "Unlimited (" << fxlongdims[i].getSize() << ")" << std::endl;
    //     else
    //    std::cout << fxlongdims[i].getSize() << std::endl;
    //  }
    //
    // These are the dimensions of the fire mesh
    // Dim 0: Time, Unlimited (360)
    // Dim: south_north_subgrid, 1150
    // Dim: west_east_subgrid, 1150

    // float sr_x = size(zsf,1)/(size(hgt,1)+1);
    // float sr_y = size(zsf,2)/(size(hgt,2)+1);
    // Then dxf=DX/sr_x, dyf=DY/sr_y

    std::vector<double> fxlong( fm_nx * fm_ny );
    std::vector<double> fxlat( fm_nx * fm_ny );
        
    wrfInputFile.getVar("FXLONG").getVar(startIdx, counts, fxlong.data());
    wrfInputFile.getVar("FXLAT").getVar(startIdx, counts, fxlat.data());
    
    // then pull the height at the nodes
    std::vector<double> fmHeight( fm_nx * fm_ny );

    wrfInputFile.getVar("ZSF").getVar(startIdx, counts, fmHeight.data());

    int sizeHGT_x = wrfInputFile.getVar("HGT").getDim(2).getSize();
    int sizeHGT_y = wrfInputFile.getVar("HGT").getDim(1).getSize();
    int sizeZSF_x = wrfInputFile.getVar("ZSF").getDim(2).getSize();
    int sizeZSF_y = wrfInputFile.getVar("ZSF").getDim(1).getSize();

    std::cout << "sizes: " << sizeHGT_x << ", " << sizeHGT_y << ", " << sizeZSF_x << ", " << sizeZSF_y << std::endl;

    // float sr_x = size(zsf,1)/(size(hgt,1)+1);
    // float sr_y = size(zsf,2)/(size(hgt,2)+1);
    float sr_x = sizeZSF_x/(sizeHGT_x+1);
    float sr_y = sizeZSF_y/(sizeHGT_y+1);

    std::cout << "sr_x, sr_y = (" << sr_x << ", " << sr_y << ")" << std::endl;

    float dxf = m_dx / sr_x;
    float dyf = m_dy / sr_y;    
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
//    if( CSLFetchBoolean( papszMetadata, GDAL_DCAP_CREATE, FALSE ) )
//        printf( "Driver %s supports Create() method.\n", pszFormat );
//    if( CSLFetchBoolean( papszMetadata, GDAL_DCAP_CREATECOPY, FALSE ) )
//        printf( "Driver %s supports CreateCopy() method.\n", pszFormat );

    GDALDataset *poDstDS;
    char **papszOptions = NULL;

    // 
    poDstDS = poDriver->Create( "WRFOut.tiff", fm_nx, fm_ny, 1, GDT_Byte, papszOptions );


    OGRSpatialReference oSRS;
    char *pszSRS_WKT = NULL;
    GDALRasterBand *poBand;

    std::vector<float> abyRaster( fm_nx * fm_ny );

    std::cout << "setting hts" << std::endl;
    for (int i=0; i<fm_nx; i++) {
        for (int j=0; j<fm_ny; j++) {
            
            int l_idx = i + j*fm_nx;
            // std::cout << "Setting ht: " << ((fmHeight[l_idx]-minHt)/rangeHt) * 255 << std::endl;
            abyRaster[ l_idx ] = ((fmHeight[l_idx];
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

    std::cout << "raster band" << std::endl;

    poBand = poDstDS->GetRasterBand(1);
    poBand->RasterIO( GF_Write, 0, 0, fm_nx, fm_ny,
                      abyRaster.data(), fm_nx, fm_ny, GDT_Byte, 0, 0 );
    
    /* Once we're done, close properly the dataset */
    GDALClose( (GDALDatasetH) poDstDS );


        //
        // Relief
        //
        NcVar reliefVar = wrfInputFile.getVar("HGT");
        std::cout << "relief dim count: " << reliefVar.getDimCount() << std::endl;
        std::vector<NcDim> dims = reliefVar.getDims();
        long totalDim = 1;
        for (int i=0; i<dims.size(); i++) {
            std::cout << "Dim: " << dims[i].getName() << ", ";
            if (dims[i].isUnlimited())
                std::cout << "Unlimited (" << dims[i].getSize() << ")" << std::endl;
            else
                std::cout << dims[i].getSize() << std::endl;
            totalDim *= dims[i].getSize();
        }
        std::cout << "relief att count: " << reliefVar.getAttCount() << std::endl;
        std::map<std::string, NcVarAtt> reliefVar_attrMap = reliefVar.getAtts();
        for (std::map<std::string, NcVarAtt>::const_iterator ci=reliefVar_attrMap.begin();
             ci!=reliefVar_attrMap.end(); ++ci) {
            std::cout << "Relief Attr: " << ci->first << std::endl;
    }

    // this is a 114 x 114 x 41 x 360 dim array...
    // slice by slice 114 x 114 per slice; 41 slices; 360 times

            // How do I extract out the start to end in and and y?
            // PHB = double(PHB(SimData.XSTART:SimData.XEND,
            // SimData.YSTART:SimData.YEND, :, SimData.TIMEVECT));

    // we want all slice data (xstart:xend X ystart:yend) at all
    // slices but only for the first 2 time series (TIMEVECT)
    double* reliefData = new double[ totalDim ];
    reliefVar.getVar( reliefData );
    
    // whole thing is in... now
    long subsetDim = nx * ny;
    relief = new double[ subsetDim ];
    for (auto l=0; l<subsetDim; l++)
        relief[l] = reliefData[l];
        
    std::cout << "first 10 for relief" << std::endl;
    for (auto l=0; l<10; l++)
        std::cout << relief[l] << std::endl;

    std::cout << "last 10 for relief" << std::endl;
    for (auto l=subsetDim-10-1; l<subsetDim; l++)
        std::cout << relief[l] << std::endl;

    delete [] reliefData;

    // % Wind data    
    // SimData = WindFunc(SimData); 
    readWindData();

    //
    // LU = ncread(WRFFile,'LU_INDEX');
    //
    NcVar LUIndexVar = wrfInputFile.getVar("LU_INDEX");
    std::cout << "LUIndex dim count: " << LUIndexVar.getDimCount() << std::endl;
    dims.clear();
    dims = LUIndexVar.getDims();
    totalDim = 1;
    for (int i=0; i<dims.size(); i++) {
        std::cout << "Dim: " << dims[i].getName() << ", ";
        if (dims[i].isUnlimited())
            std::cout << "Unlimited (" << dims[i].getSize() << ")" << std::endl;
        else
            std::cout << dims[i].getSize() << std::endl;
        totalDim *= dims[i].getSize();
    }
    std::cout << "LUIndex att count: " << LUIndexVar.getAttCount() << std::endl;
    std::map<std::string, NcVarAtt> LUIndexVar_attrMap = LUIndexVar.getAtts();
    for (std::map<std::string, NcVarAtt>::const_iterator ci=LUIndexVar_attrMap.begin();
         ci!=LUIndexVar_attrMap.end(); ++ci) {
        std::cout << "LUIndex Attr: " << ci->first << std::endl;
    }

    // this is a 114 x 114 x 41 x 360 dim array...
    // slice by slice 114 x 114 per slice; 41 slices; 360 times

    // we want all slice data (xstart:xend X ystart:yend) at all
    // slices but only for the first 2 time series (TIMEVECT)
    double* LUIndexData = new double[ totalDim ];
    LUIndexVar.getVar( LUIndexData );
    
    // whole thing is in... now
    subsetDim = nx * ny;
    double* LUIndexSave = new double[ subsetDim ];
    for (auto l=0; l<subsetDim; l++)
        LUIndexSave[l] = LUIndexData[l];
        
    std::cout << "first 10 for LUIndex" << std::endl;
    for (auto l=0; l<10; l++)
        std::cout << LUIndexSave[l] << std::endl;

    std::cout << "last 10 for LUIndex" << std::endl;
    for (auto l=subsetDim-10-1; l<subsetDim; l++)
        std::cout << LUIndexSave[l] << std::endl;

    delete [] LUIndexData;

//    }catch(NcException& e)
//     {
//       e.what();
//       cout<<"FAILURE*************************************"<<endl;
//       return;
//     }

    if (nx * ny > m_maxTerrainSize) {
        smoothDomain();
    }
#endif    
}


void WRFInput::roughnessLength()
{
#if 0
% Computes a roughness length array covering each point of the grid
% In case 1 INPUT file must be WRF RESTART file
% In case 2 INPUT file must be WRF OUTPUT
% In case 3 INPUT variable is a constant value

switch Z0Flag
    
    case 1   %%% WRF RESTART file
        
        Z0 = ncread(SimData.Z0DataSource ,'Z0');
        Z0 = Z0(SimData.XSTART:SimData.XEND, SimData.YSTART:SimData.YEND);
        
    case 2   %%% WRF OUTPUT file
        
        Z0 = zeros(size(SimData.LU,1),size(SimData.LU,2));
        
        for i = 1:size(SimData.LU,1)
            for j = 1:size(SimData.LU,2) 
                
                switch SimData.LU(i,j) 
                    
%%%%%%%%%%%%%%%%%%%%%%%%%     USGS LAND COVER     %%%%%%%%%%%%%%%%%%%%%%%%% 

%                     case 1   %%% Urban and built-up land
%                         Z0(i,j) = 0.8;
%                     case 2   %%% Dryland cropland and pasture
%                         Z0(i,j) = 0.15;
%                     case 3   %%% Irrigated cropland and pasture
%                         Z0(i,j) = 0.1;
%                     case 4   %%% Mixed dryland/irrigated cropland and pasture
%                         Z0(i,j) = 0.15;
%                     case 5   %%% Cropland/grassland mosaic
%                         Z0(i,j) = 0.14;
%                     case 6   %%% Cropland/woodland mosaic
%                         Z0(i,j) = 0.2;
%                     case 7   %%% Grassland
%                         Z0(i,j) = 0.12;
%                     case 8   %%% Shrubland
%                         Z0(i,j) = 0.05;
%                     case 9   %%% Mixed shrubland/grassland
%                         Z0(i,j) = 0.06;
%                     case 10   %%% Savanna
%                         Z0(i,j) = 0.15;
%                     case 11   %%% Deciduous broadleaf forest
%                         Z0(i,j) = 0.5;
%                     case 12   %%% Deciduous needleleaf forest
%                         Z0(i,j) = 0.5;
%                     case 13   %%% Evergreeen broadleaf forest
%                         Z0(i,j) = 0.5;
%                     case 14   %%% Evergreen needleleaf forest
%                         Z0(i,j) = 0.5;
%                     case 15   %%% Mixed forest
%                         Z0(i,j) = 0.5;
%                     case 16   %%% Water bodies
%                         Z0(i,j) = 0.0001;
%                     case 17   %%% Herbaceous wetland
%                         Z0(i,j) = 0.2;
%                     case 18   %%% Wooded wetland
%                         Z0(i,j) = 0.4;
%                     case 19   %%% Barren or sparsely vegetated
%                         Z0(i,j) = 0.01;
%                     case 20   %%% Herbaceous tundra
%                         Z0(i,j) = 0.1;
%                     case 21   %%% Wooded tundra
%                         Z0(i,j) = 0.3;
%                     case 22   %%% Mixed tundra
%                         Z0(i,j) = 0.15;
%                     case 23   %%% Bare ground tundra
%                         Z0(i,j) = 0.1;
%                     case 24   %%% Snow or ice
%                         Z0(i,j) = 0.001;

%%%%%%%%%%%%%%%%%%%%%%%%%      MODIS-WINTER      %%%%%%%%%%%%%%%%%%%%%%%%%%

                    case 1   %%% Evergreen needleleaf forest
                        Z0(i,j) = 0.5;
                    case 2   %%% Evergreeen broadleaf forest
                        Z0(i,j) = 0.5;
                    case 3   %%% Deciduous needleleaf forest
                        Z0(i,j) = 0.5;
                    case 4   %%% Deciduous broadleaf forest
                        Z0(i,j) = 0.5;
                    case 5   %%% Mixed forests
                        Z0(i,j) = 0.5;
                    case 6   %%% Closed Shrublands
                        Z0(i,j) = 0.1;
                    case 7   %%% Open Shrublands
                        Z0(i,j) = 0.1;
                    case 8   %%% Woody Savannas
                        Z0(i,j) = 0.15;
                    case 9   %%% Savannas
                        Z0(i,j) = 0.15;
                    case 10   %%% Grasslands
                        Z0(i,j) = 0.075;
                    case 11   %%% Permanent wetlands
                        Z0(i,j) = 0.3;
                    case 12   %%% Croplands
                        Z0(i,j) = 0.075;
                    case 13   %%% Urban and built-up land
                        Z0(i,j) = 0.5;
                    case 14   %%% Cropland/natural vegetation mosaic
                        Z0(i,j) = 0.065;
                    case 15   %%% Snow or ice
                        Z0(i,j) = 0.01;
                    case 16   %%% Barren or sparsely vegetated
                        Z0(i,j) = 0.065;
                    case 17   %%% Water
                        Z0(i,j) = 0.0001;
                    case 18   %%% Wooded tundra
                        Z0(i,j) = 0.15;
                    case 19   %%% Mixed tundra
                        Z0(i,j) = 0.1;
                    case 20   %%% Barren tundra
                        Z0(i,j) = 0.06;
                    case 21   %%% Lakes
                        Z0(i,j) = 0.0001;
                end
            end
        end
        
    case 3   %%% User-defined constant
        
        Z0 = repmat(SimData.Z0DataSource, SimData.XEND-SimData.XSTART+1, SimData.YEND-SimData.YSTART+1);
end
    #endif
}



void WRFInput::readWindData()
{
    // This function computes velocity magnitude, direction and vertical
    // coordinates from WRF velocity components U,V and geopotential height.
    // Values are interpolated at each corresponding cell center.

    // Extraction of the Wind data vertical position
    NcVar phbVar = wrfInputFile.getVar("PHB");
    std::cout << "PHB dim count: " << phbVar.getDimCount() << std::endl;
    std::vector<NcDim> dims = phbVar.getDims();
    long totalDim = 1;
    for (auto i=0; i<dims.size(); i++) {
        std::cout << "Dim: " << dims[i].getName() << ", ";
        if (dims[i].isUnlimited())
            std::cout << "Unlimited (" << dims[i].getSize() << ")" << std::endl;
        else
            std::cout << dims[i].getSize() << std::endl;
        totalDim *= dims[i].getSize();
    }
    std::cout << "PHB att count: " << phbVar.getAttCount() << std::endl;
    std::map<std::string, NcVarAtt> phbVar_attrMap = phbVar.getAtts();
    for (std::map<std::string, NcVarAtt>::const_iterator ci=phbVar_attrMap.begin();
         ci!=phbVar_attrMap.end(); ++ci) {
        std::cout << "PHB Attr: " << ci->first << std::endl;
    }


    // ////////////////////
    // Test reading in of this data into an array we can easily deal
    // with
#if 0 
    // WRF multi-dim format
    int timeDim = dims[0].getSize();
    int zDim = dims[1].getSize();
    int yDim = dims[2].getSize();
    int xDim = dims[3].getSize();
    
    std::cout << "PHB Dims: t=" << timeDim << "< z=" << zDim << ", y=" << yDim << ", x=" << xDim << std::endl;
    double* allPHBData = new double[ timeDim * zDim * yDim * xDim ];
    phbVar.getVar( allPHBData );

    dumpWRFDataArray("PHB", allPHBData, timeDim, zDim, yDim, xDim);
#endif
    // ////////////////////

    // this is a 114 x 114 x 41 x 360 dim array...
    // slice by slice 114 x 114 per slice; 41 slices; 360 times

    // Need to extract out the start to end in and and y?
    // PHB = double(PHB(SimData.XSTART:SimData.XEND,
    // SimData.YSTART:SimData.YEND, :, SimData.TIMEVECT));
    // can use getVar to do this.

    int nz = 41;
    long subsetDim = 2 * nz * ny * nx;

    // PHB(Time, bottom_top_stag, south_north, west_east) ;
    double* phbData = new double[ subsetDim ];
    std::vector< size_t > starts = { 0, 0, 0, 0 };
    std::vector< size_t > counts = { 2, 41, 114, 114 };   // depends on order of dims
    phbVar.getVar( starts, counts, phbData );
                     
    dumpWRFDataArray("PHB Subset", phbData, 2, 41, 114, 114);

    // 
    // Extraction of the Wind data vertical position
    // 
    NcVar phVar = wrfInputFile.getVar("PH");

    double* phData = new double[ subsetDim ];
    phVar.getVar( starts, counts, phData );
    

    // 
    /// Height
    // 
    double* heightData = new double[ subsetDim ];
    for (auto l=0; l<subsetDim; l++) {
        heightData[l] = (phbData[l] + phData[l]) / 9.81;
    }
    
    // Extraction of the Ustagg
    // Ustagg = ncread(SimData.WRFFile,'U');
    // Ustagg = Ustagg(SimData.XSTART:SimData.XEND +1, SimData.YSTART:SimData.YEND, :, SimData.TIMEVECT);
    NcVar uStaggered = wrfInputFile.getVar("U");

    std::vector<NcDim> ustagg_dims = uStaggered.getDims();
    for (int i=0; i<ustagg_dims.size(); i++) {
        std::cout << "Dim: " << ustagg_dims[i].getName() << ", ";
        if (ustagg_dims[i].isUnlimited())
            std::cout << "Unlimited (" << ustagg_dims[i].getSize() << ")" << std::endl;
        else
            std::cout << ustagg_dims[i].getSize() << std::endl;
    }
    
    // time, Z, Y, X is order
    starts.clear(); counts.clear();
    starts = { 0, 0, 0, 0 };
    counts = { 2, 40, 114, 115 };
    subsetDim = 1;
    for (auto i=0; i<counts.size(); i++)  {
        subsetDim *= (counts[i] - starts[i]);
    }
    
    double* uStaggeredData = new double[ subsetDim ];
    uStaggered.getVar( starts, counts, uStaggeredData );
    dumpWRFDataArray("Ustagg", uStaggeredData, 2, 40, 114, 115);

    // 
    // Vstagg = ncread(SimData.WRFFile,'V');
    // Vstagg = Vstagg(SimData.XSTART:SimData.XEND, SimData.YSTART:SimData.YEND +1, :, SimData.TIMEVECT);
    //
    NcVar vStaggered = wrfInputFile.getVar("V");
    
    starts.clear();  counts.clear();
    starts = { 0, 0, 0, 0 };
    counts = { 2, 40, 115, 114 };
    subsetDim = 1;
    for (auto i=0; i<counts.size(); i++) 
        subsetDim *= (counts[i] - starts[i]);
    
    double* vStaggeredData = new double[ subsetDim ];
    vStaggered.getVar( starts, counts, vStaggeredData );

    
    // 
    // %% Centering values %%
    // SimData.NbAlt = size(Height,3) - 1;
    //
    int nbAlt = 40;  // zDim - 1 but hack for now -- need to be computed
    
    // ///////////////////////////////////
    // U = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
    // for x = 1:SimData.nx
    //   U(x,:,:,:) = .5*(Ustagg(x,:,:,:) + Ustagg(x+1,:,:,:));
    // end
    // ///////////////////////////////////    

    // Just make sure we've got the write dims here
    nx = 114;
    ny = 114;
    
    std::vector<double> U( nx * ny * nbAlt * 2, 0.0 );
    for (auto t=0; t<2; t++) {
        for (auto z=0; z<nbAlt; z++) {
            for (auto y=0; y<ny; y++) {
                for (auto x=0; x<nx; x++) {

                    auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
                    auto idxP1x = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + (x+1);

                    U[idx] = 0.5 * ( uStaggeredData[idx] + uStaggeredData[idxP1x] );
                }
            }
        }
    }
    dumpWRFDataArray("U", U.data(), 2, 40, 114, 114);
    
    // V = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
    // for y = 1:SimData.ny
    //    V(:,y,:,:) = .5*(Vstagg(:,y,:,:) + Vstagg(:,y+1,:,:));
    // end
    std::vector<double> V( nx * ny * nbAlt * 2, 0.0 );
    for (auto t=0; t<2; t++) {
        for (auto z=0; z<nbAlt; z++) {
            for (auto y=0; y<ny; y++) {
                for (auto x=0; x<nx; x++) {
                    
                    auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
                    auto idxP1y = t * (nbAlt * ny * nx) + z * (ny * nx) + (y+1) * (nx) + x;

                    V[idx] = 0.5 * ( vStaggeredData[idx] + vStaggeredData[idxP1y] );
                }
            }
        }
    }
    dumpWRFDataArray("V", V.data(), 2, 40, 114, 114);
    
    // SimData.CoordZ = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
    // for k = 1:SimData.NbAlt
    //    SimData.CoordZ(:,:,k,:) = .5*(Height(:,:,k,:) + Height(:,:,k+1,:));
    // end
    std::vector<double> coordZ( nx * ny * nbAlt * 2, 0.0 );
    for (auto t=0; t<2; t++) {
        for (auto z=0; z<nbAlt; z++) {
            for (auto y=0; y<ny; y++) {
                for (auto x=0; x<nx; x++) {

                    auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
                    auto idxP1z = t * (nbAlt * ny * nx) + (z+1) * (ny * nx) + y * (nx) + x;

                    coordZ[idx] = 0.5 * ( heightData[idx] + heightData[idxP1z] );
                }
            }
        }
    }

    // %% Velocity and direction %%
    // SimData.WS = sqrt(U.^2 + V.^2);
    std::vector<double> WS( nx * ny * nbAlt * 2, 0.0 );
    for (auto t=0; t<2; t++) {
        for (auto z=0; z<nbAlt; z++) {
            for (auto y=0; y<ny; y++) {
                for (auto x=0; x<nx; x++) {
                    auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
                    WS[idx] = sqrt( U[idx]*U[idx] + V[idx]*V[idx] );
                }
            }
        }

        
    }
    dumpWRFDataArray("WS", WS.data(), 2, nbAlt, ny, nx);
    

    // SimData.WD = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
    std::vector<double> WD( nx * ny * nbAlt * 2, 0.0 );
    for (auto t=0; t<2; t++) {
        for (auto z=0; z<nbAlt; z++) {
            for (auto y=0; y<ny; y++) {
                for (auto x=0; x<nx; x++) {
                    auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
                    if (U[idx]> 0) {
                        WD[idx] = 270.0 - (180.0/c_PI) * atan( V[idx]/U[idx] );
                    }
                    else {
                        WD[idx] = 90.0 - (180.0/c_PI) * atan( V[idx]/U[idx] );
                    }
                }
            }
        }
    }
    dumpWRFDataArray("WD", WD.data(), 2, nbAlt, ny, nx);

#if 0
I shoould need this... hopefully
%% Permutation to set dimensions as (row,col,etc) = (ny,nx,nz,nt) %%

SimData.WD = permute(SimData.WD,[2 1 3 4]);
SimData.WS = permute(SimData.WS,[2 1 3 4]);
SimData.CoordZ = permute(SimData.CoordZ,[2 1 3 4]);
#endif

    std::cout << "Smoothing domain" << std::endl;

    // Updating dimensions
    int nx2 = floor(sqrt(m_maxTerrainSize * nx/(float)ny));
    int ny2 = floor(sqrt(m_maxTerrainSize * ny/(float)nx));
    
    // SimData.nx = nx2;
    // SimData.ny = ny2;
    m_dx = m_dx * nx/(float)nx2;
    m_dy = m_dy * ny/(float)ny2;

    std::cout << "Resizing from " << nx << " by " << ny << " to " << nx2 << " by " << ny2 << std::endl;

    // Terrain
    // SimData.Relief = imresize(SimData.Relief,[ny2,nx2]);
    std::vector<double> reliefResize( nx2 * ny2 * nbAlt * 2, 0.0 );

    //
    // Fake this for now with nearest neighbor... implement bicubic
    // later
    //

    float scaleX = nx / (float)nx2;
    float scaleY = ny / (float)ny2;
    
    for (auto y=0; y<ny2; y++) {
        for (auto x=0; x<nx2; x++) {
                    
            // Map this into the larger vector
            int largerX = (int)floor( x * scaleX );
            int largerY = (int)floor( y * scaleY );
            
            auto idxLarger = largerY * (nx) + largerX;
            auto idx = y * (nx2) + x;

            reliefResize[idx] = relief[idxLarger];
        }
    }


}

void WRFInput::setWRFDataPoint()
{
#if 0
% If MaxNbStat is smaller than the number of WRF data point available, then
% we operate a selection. Vertical height is selected between defined boundaries.
% For each stations, their horizontal and vertical coordinates, wind speed 
% and direction, along with the number of vertical altitude pts, are recorded 

if SimData.nx*SimData.ny > SimData.MaxNbStat
    
    nx2 = sqrt(SimData.MaxNbStat*SimData.nx/SimData.ny);
    ny2 = sqrt(SimData.MaxNbStat*SimData.ny/SimData.nx);
    
    WRF_RowY = (1:SimData.ny/ny2:SimData.ny);
    WRF_ColX = (1:SimData.nx/nx2:SimData.nx);
    
    WRF_RowY = unique(round(WRF_RowY));
    WRF_ColX = unique(round(WRF_ColX));
    
else
    
    WRF_RowY = (1:SimData.ny);
    WRF_ColX = (1:SimData.nx);
    
end

SimData.NbStat = numel(WRF_RowY)*numel(WRF_ColX);
StatData.CoordX = zeros(1,SimData.NbStat);
StatData.CoordY = zeros(1,SimData.NbStat);
StatData.nz = zeros(1,SimData.NbStat);

StatData.CoordZ = struct([]);
StatData.WS = struct([]);
StatData.WD = struct([]);


Stat = 1;
for y = WRF_RowY
    for x = WRF_ColX
        
        StatData.CoordX(Stat) = x;
        StatData.CoordY(Stat) = y;
        
        levelk_max = 0;
        for t = 1:numel(SimData.TIMEVECT)                     
            CoordZ_xyt = SimData.CoordZ(y,x,:,t);        
            [levelk] = find(CoordZ_xyt >= SimData.MinWRFAlt & CoordZ_xyt <= SimData.MaxWRFAlt);
            if numel(levelk) > numel(levelk_max)
                levelk_max = levelk; % If wind data heights change during time, higher height vector is selected
            end
        end
        
        StatData.CoordZ{Stat} = reshape(SimData.CoordZ(y,x,levelk_max,:),numel(levelk_max),size(SimData.CoordZ,4));
        StatData.nz(Stat) = size(StatData.CoordZ{Stat},1);
        
        StatData.WS{Stat} = reshape(SimData.WS(y,x,levelk_max,:),numel(levelk_max),size(SimData.WS,4));
        StatData.WD{Stat} = reshape(SimData.WD(y,x,levelk_max,:),numel(levelk_max),size(SimData.WD,4));   
        
        Stat = Stat + 1;
    end
end

SimData.maxCoordz = 0; % Will be used to set domain vertical dimension
SimData.minCoordz = 0;
for i = 1:SimData.NbStat
    SimData.minCoordz = min(min(min(SimData.maxCoordz,StatData.CoordZ{i})));
    SimData.maxCoordz = max(max(max(SimData.maxCoordz,StatData.CoordZ{i})));
end

fprintf('%i %s %g %s %g %s\n',SimData.NbStat,' WRF data points have been generated between ',SimData.minCoordz,' and ',SimData.maxCoordz,' meters AGL')
#endif


}


void WRFInput::dumpWRFDataArray(const std::string &name, double *data, int dimT, int dimZ, int dimY, int dimX)
{
    std::cout << "[" << name << "] WRF Data Dump" << std::endl << "==========================" << std::endl;

    // This output is analagous to Matlab's Columns 1 through dimY style of output
    for (auto t=0; t<dimT; t++) {
        for (auto z=0; z<dimZ; z++) {
            std::cout << "Slice: (t=" << t << ", z=" << z << ")" << std::endl;
            for (auto x=0; x<dimX; x++) {
                for (auto y=0; y<dimY; y++) {

                    auto idx = t * (dimZ * dimY * dimX) + z * (dimY * dimX) + y * (dimX) + x;
                    std::cout << data[idx] << ' ';
                    
                }
                std::cout << std::endl;
            }
        }
    }
}


void WRFInput::smoothDomain()
{
    // Smooth the following layers: topography, wind data, Z0 and
    // land-use using imresize (bicubic interpolation by default).
    // Land-use is interpolated with the "nearest point" method as we
    // must not change the categories values.
    
#if 0
    std::cout << "Smoothing domain" << std::endl;

    // Updating dimensions
    int nx2 = floor(sqrt(m_maxTerrainSize * nx/(float)ny));
    int ny2 = floor(sqrt(m_maxTerrainSize * ny/(float)nx));
    
    // SimData.nx = nx2;
    // SimData.ny = ny2;
    m_dx = m_dx * nx/(float)nx2;
    m_dy = m_dy * ny/(float)ny2;

    // Terrain
    // SimData.Relief = imresize(SimData.Relief,[ny2,nx2]);
    int nbAlt = 40;
    std::vector<double> reliefResize( nx_2 * ny_2 * nbAlt * 2, 0.0 );

    // fake this for now with nearest neighbor...
    for (auto t=0; t<2; t++) {
        for (auto z=0; z<nbAlt; z++) {

            for (auto y=0; y<ny2; y++) {
                for (auto x=0; x<nx2; x++) {
                    
                    // Map this into the larger vector

                }
            }

        }
    }
    
            
#endif



    // Wind velocity, direction and vertical position
    // SimData.WS = imresize(SimData.WS,[ny2,nx2]);
    // SimData.WD = imresize(SimData.WD,[ny2,nx2]);
    // SimData.CoordZ = imresize(SimData.CoordZ,[ny2,nx2]);

    // Roughness length
    // SimData.Z0 = imresize(SimData.Z0,[ny2,nx2]);

    // Land-use
    // SimData.LU = imresize(SimData.LU,[ny2,nx2],'nearest');

}


void WRFInput::minimizeDomainHeight()
{
    // Here we lower minimum altitude to 0 in order to save
    // computational space.  For the same reason, topography below 50
    // cm will not be considered.

    // SimData.OldTopoMin = min(min(SimData.Relief));
    // SimData.Relief = SimData.Relief - SimData.OldTopoMin;  % Lowering minimum altitude to 0 

    // IndLowRelief = find(SimData.Relief <= 15); % Relief below 0.5m is not considered
    // SimData.Relief(IndLowRelief) = 0;

    // SimData.NbTerrain = numel(SimData.Relief) - size(IndLowRelief,1);
    // SimData.NewTopoMax = max(max(SimData.Relief));
    // SimData.CoordZ = SimData.CoordZ - SimData.OldTopoMin;

}
