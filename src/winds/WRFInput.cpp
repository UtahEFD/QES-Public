
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
#include "cpl_conv.h"// for CPLMalloc()

#include "WINDSGeneralData.h"

void UTMConv(double &rlon, double &rlat, double &rx, double &ry, int &UTM_PROJECTION_ZONE, int iway)
{


  /*

                  S p e c f e m 3 D  V e r s i o n  2 . 1
                  ---------------------------------------

             Main authors: Dimitri Komatitsch and Jeroen Tromp
       Princeton University, USA and CNRS / INRIA / University of Pau
    (c) Princeton University / California Institute of Technology and CNRS / INRIA / University of Pau
                                July 2012

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

  */

  /*
    UTM (Universal Transverse Mercator) projection from the USGS
  */


  /*
  convert geodetic longitude and latitude to UTM, and back
  use iway = ILONGLAT2UTM for long/lat to UTM, IUTM2LONGLAT for UTM to lat/long
  a list of UTM zones of the world is available at www.dmap.co.uk/utmworld.htm
  */


  /*
        CAMx v2.03

        UTM_GEO performs UTM to geodetic (long/lat) translation, and back.

        This is a Fortran version of the BASIC program "Transverse Mercator
        Conversion", Copyright 1986, Norman J. Berls (Stefan Musarra, 2/94)
        Based on algorithm taken from "Map Projections Used by the USGS"
        by John P. Snyder, Geological Survey Bulletin 1532, USDI.

        Input/Output arguments:

           rlon                  Longitude (deg, negative for West)
           rlat                  Latitude (deg)
           rx                    UTM easting (m)
           ry                    UTM northing (m)
           UTM_PROJECTION_ZONE  UTM zone
           iway                  Conversion type
                                 ILONGLAT2UTM = geodetic to UTM
                                 IUTM2LONGLAT = UTM to geodetic
  */


  int ILONGLAT2UTM = 0, IUTM2LONGLAT = 1;
  float PI = 3.141592653589793;
  float degrad = PI / 180.0;
  float raddeg = 180.0 / PI;
  float semimaj = 6378206.40;
  float semimin = 6356583.80;
  float scfa = 0.99960;

  /*
    some extracts about UTM:

    There are 60 longitudinal projection zones numbered 1 to 60 starting at 180Â°W.
    Each of these zones is 6 degrees wide, apart from a few exceptions around Norway and Svalbard.
    There are 20 latitudinal zones spanning the latitudes 80Â°S to 84Â°N and denoted
    by the letters C to X, ommitting the letter O.
    Each of these is 8 degrees south-north, apart from zone X which is 12 degrees south-north.

    To change the UTM zone and the hemisphere in which the
    calculations are carried out, need to change the fortran code and recompile. The UTM zone is described
    actually by the central meridian of that zone, i.e. the longitude at the midpoint of the zone, 3 degrees
    from either zone boundary.
    To change hemisphere need to change the "north" variable:
    - north=0 for northern hemisphere and
    - north=10000000 (10000km) for southern hemisphere. values must be in metres i.e. north=10000000.

    Note that the UTM grids are actually Mercators which
    employ the standard UTM scale factor 0.9996 and set the
    Easting Origin to 500,000;
    the Northing origin in the southern
    hemisphere is kept at 0 rather than set to 10,000,000
    and this gives a uniform scale across the equator if the
    normal convention of selecting the Base Latitude (origin)
    at the equator (0 deg.) is followed.  Northings are
    positive in the northern hemisphere and negative in the
    southern hemisphere.
    */

  float north = 0.0;
  float east = 500000.0;

  float e2, e4, e6, ep2, xx, yy, dlat, dlon, zone, cm, cmr, delam;
  float f1, f2, f3, f4, rm, rn, t, c, a, e1, u, rlat1, dlat1, c1, t1, rn1, r1, d;
  double rx_save, ry_save, rlon_save, rlat_save;

  // save original parameters
  rlon_save = rlon;
  rlat_save = rlat;
  rx_save = rx;
  ry_save = ry;

  xx = 0.0;
  yy = 0.0;
  dlat = 0.0;
  dlon = 0.0;

  // define parameters of reference ellipsoid
  e2 = 1.0 - pow((semimin / semimaj), 2.0);
  e4 = pow(e2, 2.0);
  e6 = e2 * e4;
  ep2 = e2 / (1.0 - e2);

  if (iway == IUTM2LONGLAT) {
    xx = rx;
    yy = ry;
  } else {
    dlon = rlon;
    dlat = rlat;
  }

  // Set Zone parameters

  zone = UTM_PROJECTION_ZONE;
  // sets central meridian for this zone
  cm = zone * 6.0 - 183.0;
  cmr = cm * degrad;

  // Lat/Lon to UTM conversion

  if (iway == ILONGLAT2UTM) {
    rlon = degrad * dlon;
    rlat = degrad * dlat;

    delam = dlon - cm;
    if (delam < -180.0) {
      delam = delam + 360.0;
    }
    if (delam > 180.0) {
      delam = delam - 360.0;
    }
    delam = delam * degrad;

    f1 = (1.0 - (e2 / 4.0) - 3.0 * (e4 / 64.0) - 5.0 * (e6 / 256)) * rlat;
    f2 = 3.0 * (e2 / 8.0) + 3.0 * (e4 / 32.0) + 45.0 * (e6 / 1024.0);
    f2 = f2 * sin(2.0 * rlat);
    f3 = 15.0 * (e4 / 256.0) * 45.0 * (e6 / 1024.0);
    f3 = f3 * sin(4.0 * rlat);
    f4 = 35.0 * (e6 / 3072.0);
    f4 = f4 * sin(6.0 * rlat);
    rm = semimaj * (f1 - f2 + f3 - f4);
    if (dlat == 90.0 || dlat == -90.0) {
      xx = 0.0;
      yy = scfa * rm;
    } else {
      rn = semimaj / sqrt(1.0 - e2 * pow(sin(rlat), 2.0));
      t = pow(tan(rlat), 2.0);
      c = ep2 * pow(cos(rlat), 2.0);
      a = cos(rlat) * delam;

      f1 = (1.0 - t + c) * pow(a, 3.0) / 6.0;
      f2 = 5.0 - 18.0 * t + pow(t, 2.0) + 72.0 * c - 58.0 * ep2;
      f2 = f2 * pow(a, 5.0) / 120.0;
      xx = scfa * rn * (a + f1 + f2);
      f1 = pow(a, 2.0) / 2.0;
      f2 = 5.0 - t + 9.0 * c + 4.0 * pow(c, 2.0);
      f2 = f2 * pow(a, 4.0) / 24.0;
      f3 = 61.0 - 58.0 * t + pow(t, 2.0) + 600.0 * c - 330.0 * ep2;
      f3 = f3 * pow(a, 6.0) / 720.0;
      yy = scfa * (rm + rn * tan(rlat) * (f1 + f2 + f3));
    }
    xx = xx + east;
    yy = yy + north;
  }

  // UTM to Lat/Lon conversion

  else {
    xx = xx - east;
    yy = yy - north;
    e1 = sqrt(1.0 - e2);
    e1 = (1.0 - e1) / (1.0 + e1);
    rm = yy / scfa;
    u = 1.0 - (e2 / 4.0) - 3.0 * (e4 / 64.0) - 5.0 * (e6 / 256.0);
    u = rm / (semimaj * u);

    f1 = 3.0 * (e1 / 2.0) - 27.0 * pow(e1, 3.0) / 32.0;
    f1 = f1 * sin(2.0 * u);
    f2 = (21.0 * pow(e1, 2.0) / 16.0) - 55.0 * pow(e1, 4.0) / 32.0;
    f2 = f2 * sin(4.0 * u);
    f3 = 151.0 * pow(e1, 3.0) / 96.0;
    f3 = f3 * sin(6.0 * u);
    rlat1 = u + f1 + f2 + f3;
    dlat1 = rlat1 * raddeg;
    if (dlat1 >= 90.0 || dlat1 <= -90.0) {
      dlat1 = std::min(dlat1, 90.0f);
      dlat1 = std::max(dlat1, -90.0f);
      dlon = cm;
    } else {
      c1 = ep2 * pow(cos(rlat1), 2.0);
      t1 = pow(tan(rlat1), 2.0);
      f1 = 1.0 - e2 * pow(sin(rlat1), 2.0);
      rn1 = semimaj / sqrt(f1);
      r1 = semimaj * (1.0 - e2) / sqrt(pow(f1, 3.0));
      d = xx / (rn1 * scfa);

      f1 = rn1 * tan(rlat1) / r1;
      f2 = pow(d, 2.0) / 2.0;
      f3 = 5.0 * 3.0 * t1 + 10.0 * c1 - 4.0 * pow(c1, 2.0) - 9.0 * ep2;
      f3 = f3 * pow(d, 2.0) * pow(d, 2.0) / 24.0;
      f4 = 61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * pow(t1, 2.0) - 252.0 * ep2 - 3.0 * pow(c1, 2.0);
      f4 = f4 * pow(pow(d, 2.0), 3.0) / 720.0;
      rlat = rlat1 - f1 * (f2 - f3 + f4);
      dlat = rlat * raddeg;

      f1 = 1.0 + 2.0 * t1 + c1;
      f1 = f1 * pow(d, 2.0) * d / 6.0;
      f2 = 5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * pow(c1, 2.0) + 8.0 * ep2 + 24.0 * pow(t1, 2.0);
      f2 = f2 * pow(pow(d, 2.0), 2.0) * d / 120.0;
      rlon = cmr + (d - f1 + f2) / cos(rlat1);
      dlon = rlon * raddeg;
      if (dlon < -180.0) {
        dlon = dlon + 360.0;
      }
      if (dlon > 180.0) {
        dlon = dlon - 360.0;
      }
    }
  }

  if (iway == IUTM2LONGLAT) {
    rlon = dlon;
    rlat = dlat;
    rx = rx_save;
    ry = ry_save;
  } else {
    rx = xx;
    ry = yy;
    rlon = rlon_save;
    rlat = rlat_save;
  }
}


int checksum(int currChkSum, std::vector<float> &data)
{
  union equivalence {
    int i;
    float r;
  };


  for (auto idx = 0u; idx < data.size(); idx++) {
    equivalence num;
    num.r = data[idx];
    currChkSum = currChkSum ^ num.i;
  }
  return currChkSum;
}


WRFInput::WRFInput(const std::string &filename,
                   double domainUTMx,
                   double domainUTMy,
                   int zoneUTM,
                   std::string &zoneLetterUTM,
                   float dimX,
                   float dimY,
                   int sensorSample,
                   bool performWRFCoupling,
                   bool sensorsOnly)
  : m_WRFFilename(filename),
    m_processOnlySensorData(sensorsOnly),
    wrfInputFile(filename, NcFile::write),
    m_minWRFAlt(22), m_maxWRFAlt(330), m_maxTerrainSize(10001), m_maxNbStat(156),
    m_TerrainFlag(1), m_BdFlag(0), m_VegFlag(0), m_Z0Flag(2),
    m_performWRFRunCoupling(performWRFCoupling)
{
  std::cout << "WRF Input Processor - reading data from " << filename << std::endl;
  if (sensorsOnly) {
    std::cout << "\tOnly parsing wind velocity profiles from WRF file." << std::endl;
  }

  if (m_performWRFRunCoupling) {
    std::cout << "\n============> WRF-QES Run Coupling Enabled!\n"
              << std::endl;
  }

  // How is UTM used now?  --Pete
  int UTMZone = zoneUTM;

  // Acquire some global attributes from the WRF system
  std::multimap<std::string, NcGroupAtt> globalAttributes = wrfInputFile.getAtts();

  // Extract the WRF version information
  std::string wrfTitle;
  auto gblAttIter = globalAttributes.find("TITLE");
  gblAttIter->second.getValues(wrfTitle);

  std::size_t vLoc = 0, nextSpace = 0;
  vLoc = wrfTitle.find(" V");
  std::string subStr1 = wrfTitle.substr(vLoc + 2, wrfTitle.length() - 1);
  nextSpace = subStr1.find(" ");
  std::string vString = subStr1.substr(0, nextSpace);

  // Wait until WRF has run and placed the correct checksum and timestamp number in the file
  // CHSUM0_FMW
  // FRAME0_FMW
  // int wrfCHSUM0_FMW = 0;
  int wrfFRAME0_FMW = -1;

  NcDim fmwdim = wrfInputFile.getVar("U0_FMW").getDim(0);
  int fmwTimeSize = fmwdim.getSize();

  std::vector<size_t> fmw_StartIdx = { fmwTimeSize - 1 };
  std::vector<size_t> fmw_counts = { 1 };
  wrfInputFile.getVar("FRAME0_FMW").getVar(fmw_StartIdx, fmw_counts, &wrfFRAME0_FMW);

  if (m_performWRFRunCoupling) {

    while (wrfFRAME0_FMW == -1) {
      std::cout << "Waiting for FRAME0_FMW to be initialized..." << std::endl;
      usleep(1000000);// 1 sec
      wrfInputFile.getVar("FRAME0_FMW").getVar(fmw_StartIdx, fmw_counts, &wrfFRAME0_FMW);
    }

    std::cout << "WRF-QES Coupling: Frame = " << wrfFRAME0_FMW << std::endl;
    currWRFFRAME0Num = wrfFRAME0_FMW;
  }

  // Report the version information
  std::cout << "\tWRF Version: " << vString << std::endl;

  // Grab the Stored end of the dimension and subtract 1.
  // xDim+1 is a pointer reference to the 2nd array value.
  // Same happens for yDim below.
  int xDim[2] = { 1, 0 }, yDim[2] = { 1, 0 };

  // Atmospheric mesh size is stored in GRID_DIMENSION variables
  gblAttIter = globalAttributes.find("WEST-EAST_GRID_DIMENSION");
  gblAttIter->second.getValues(xDim + 1);
  xDim[1] -= 1;

  gblAttIter = globalAttributes.find("SOUTH-NORTH_GRID_DIMENSION");
  gblAttIter->second.getValues(yDim + 1);
  yDim[1] -= 1;

  // Compute nx and ny from the Atmospheric Mesh of WRF
  atm_nx = xDim[1] - xDim[0] + 1;
  atm_ny = yDim[1] - yDim[0] + 1;

  std::cout << "WRF Atmospheric Mesh Domain is " << atm_nx << " X " << atm_ny << " cells." << std::endl;

  // Extract DX and DY of the Atm Mesh
  double cellSize[2] = { 1, 1 };
  gblAttIter = globalAttributes.find("DX");
  gblAttIter->second.getValues(cellSize);

  gblAttIter = globalAttributes.find("DY");
  gblAttIter->second.getValues(cellSize + 1);

  atm_dx = cellSize[0];
  atm_dy = cellSize[1];

  std::cout << "WRF Atmospheric Mesh Resolution (dx,dy) is (" << atm_dx << ", " << atm_dy << ")" << std::endl;

  // Check to see if both sensor data and terrain mesh data are
  // needing to be extracted
  if (m_processOnlySensorData == false) {

    // Extract BOTH "fire" mesh data and the Wind Profiles from
    // the Atmosperic Mesh

    std::cout << "Attempting to read fire mesh from WRF output file data..." << std::endl;

    //
    // IMPORTANT -- need to include code to check for fire mesh
    // data!!!
    //
    // Check to verify the fields for the fire mesh exist --
    // otherwise exit

    // Need something like this...
    // if ( !(wrfInputFile.getVar("FXLONG") && wrfInputFile.getVar("FXLAT")) ) {
    // std::cerr << "ERROR!  WRF input for reading terrain requires that Fire Mesh data exists in WRF Output file." << std::endl;
    // std::cerr << "Exiting." << std::endl;
    // exit(EXIT_FAILURE);
    // }


    //
    // Fire Mesh Terrain Nodes
    //
    // int fm_nt = wrfInputFile.getVar("FXLONG").getDim(0).getSize();
    fm_ny = wrfInputFile.getVar("FXLONG").getDim(1).getSize();
    fm_nx = wrfInputFile.getVar("FXLONG").getDim(2).getSize();

    std::vector<size_t> startIdx = { 0, 0, 0, 0 };
    std::vector<size_t> counts = { 1,
                                   static_cast<unsigned long>(fm_ny),
                                   static_cast<unsigned long>(fm_nx) };

    std::vector<double> fxlong(fm_nx * fm_ny);
    std::vector<double> fxlat(fm_nx * fm_ny);
    wrfInputFile.getVar("FXLONG").getVar(startIdx, counts, fxlong.data());
    wrfInputFile.getVar("FXLAT").getVar(startIdx, counts, fxlat.data());

    std::cout << "\treading data for FWH, FZ0, and ZSF..." << std::endl;

    std::vector<double> fwh(fm_nx * fm_ny);
    std::vector<double> fz0(fm_nx * fm_ny);
    wrfInputFile.getVar("FWH").getVar(startIdx, counts, fwh.data());
    wrfInputFile.getVar("FZ0").getVar(startIdx, counts, fz0.data());

    fmHeight.resize(fm_nx * fm_ny);
    wrfInputFile.getVar("ZSF").getVar(startIdx, counts, fmHeight.data());

    // The following is provided from Jan and students
    int sizeHGT_x = wrfInputFile.getVar("HGT").getDim(2).getSize();
    int sizeHGT_y = wrfInputFile.getVar("HGT").getDim(1).getSize();
    int sizeZSF_x = wrfInputFile.getVar("ZSF").getDim(2).getSize();
    int sizeZSF_y = wrfInputFile.getVar("ZSF").getDim(1).getSize();

    float sr_x = sizeZSF_x / (sizeHGT_x + 1);
    float sr_y = sizeZSF_y / (sizeHGT_y + 1);


    fm_dx = atm_dx / sr_x;
    fm_dy = atm_dy / sr_y;

    std::cout << "Full Fire Mesh Size with border: " << fm_nx << " X " << fm_ny << std::endl;

    fm_nx = fm_nx - sr_x;
    fm_ny = fm_ny - sr_y;

    // Then dxf=DX/sr_x, dyf=DY/sr_y

    // for now, set dz = 1
    fm_dz = 1.0f;

    std::cout << "WRF Fire Mesh dimensions (nx, ny): " << fm_nx << " by " << fm_ny << std::endl;
    std::cout << "WRF Fire Mesh Resolution (dx, dy): (" << fm_dx << ", " << fm_dy << ")" << std::endl;

    std::cout << "\ttotal domain size: " << fm_nx * fm_dx << "m by " << fm_ny * fm_dy << "m" << std::endl;

    double fm_minWRFAlt = std::numeric_limits<double>::max(),
           fm_maxWRFAlt = std::numeric_limits<double>::min();

    // Scan the fire mesh to determine height ranges
    for (int i = 0; i < fm_nx; i++) {
      for (int j = 0; j < fm_ny; j++) {

        int l_idx = i + j * fm_nx;

        if (fmHeight[l_idx] > fm_maxWRFAlt) fm_maxWRFAlt = fmHeight[l_idx];
        if (fmHeight[l_idx] < fm_minWRFAlt) fm_minWRFAlt = fmHeight[l_idx];
      }
    }

    std::cout << "Terrain Min Ht: " << fm_minWRFAlt << ", Max Ht: " << fm_maxWRFAlt << std::endl;

    // if specified, use the max height... otherwise, pick one
    // that's about twice the max height to give room for the flow
    fm_nz = (int)ceil(fm_maxWRFAlt * 2.0);// this ONLY works
                                          // if dz=1.0

    std::cout << "Domain nz: " << fm_nz << std::endl;

    //
    // override any zone information?  Need to fix these use cases
    //
    UTMZone = (int)floor((fxlong[0] + 180) / 6) + 1;
    std::cout << "UTM Zone: " << UTMZone << std::endl;

    std::cout << "(Lat,Long) at Lower Left (LL) = " << fxlat[0] << ", " << fxlong[0] << std::endl;
    std::cout << "(Lat,Long) at Lower Right (LR) = " << fxlat[fm_nx - 1] << ", " << fxlong[fm_nx - 1] << std::endl;
    std::cout << "(Lat,Long) at Upper Left (UL) = " << fxlat[(fm_ny - 1) * fm_nx] << ", " << fxlong[(fm_ny - 1) * fm_nx] << std::endl;
    std::cout << "(Lat,Long) at Upper Right (UR) = " << fxlat[fm_nx - 1 + (fm_ny - 1) * fm_nx] << ", " << fxlong[fm_nx - 1 + (fm_ny - 1) * fm_nx] << std::endl;


    // variables to hold the UTM coordinates of the corners
    //   domainUTMx, domainUTMy contain the lower left origin UTM
    //   and are already defined
    double lrUTMx, lrUTMy,
      ulUTMx, ulUTMy,
      urUTMx, urUTMy;

    UTMConv(fxlong[0], fxlat[0], domainUTMx, domainUTMy, UTMZone, 0);
    std::cout << std::setprecision(9) << "\tConverted LL UTM: " << domainUTMx << ", " << domainUTMy << std::endl;

    UTMConv(fxlong[fm_nx - 1], fxlat[fm_nx - 1], lrUTMx, lrUTMy, UTMZone, 0);
    std::cout << std::setprecision(9) << "\tConverted LR UTM: " << lrUTMx << ", " << lrUTMy << std::endl;

    UTMConv(fxlong[(fm_ny - 1) * fm_nx], fxlat[(fm_ny - 1) * fm_nx], ulUTMx, ulUTMy, UTMZone, 0);
    std::cout << std::setprecision(9) << "\tConverted UL UTM: " << ulUTMx << ", " << ulUTMy << std::endl;

    UTMConv(fxlong[fm_nx - 1 + (fm_ny - 1) * fm_nx], fxlat[fm_nx - 1 + (fm_ny - 1) * fm_nx], urUTMx, urUTMy, UTMZone, 0);
    std::cout << std::setprecision(9) << "\tConverted UR UTM: " << urUTMx << ", " << urUTMy << std::endl;


    dimX = fm_nx * fm_dx;
    dimY = fm_ny * fm_dy;

    //
    // Need this anymore? ???
    std::vector<float> abyRaster(fm_nx * fm_ny);
    for (int i = 0; i < fm_nx; i++) {
      for (int j = 0; j < fm_ny; j++) {

        int l_idx = i + j * fm_nx;
        abyRaster[l_idx] = fmHeight[l_idx];
        // std::cout << "height = " << fmHeight[ l_idx ] << std::endl;
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

    //
    // this may not be needed anymore...
    // BEGIN
    double clat, clon;
    gblAttIter = globalAttributes.find("CEN_LAT");
    gblAttIter->second.getValues(&clat);

    gblAttIter = globalAttributes.find("CEN_LON");
    gblAttIter->second.getValues(&clon);

    double lon2eastings[1] = { clon };
    double lat2northings[1] = { clat };

    int srx = int(fm_nx / (atm_nx + 1));
    int sry = int(fm_ny / (atm_ny + 1));

    int nx_fire = fm_nx - srx;
    int ny_fire = fm_ny - sry;

    float dx_fire = atm_dx / (float)srx;
    float dy_fire = atm_dy / (float)sry;

    double t_x0_fire = -nx_fire / 2. * dx_fire + lon2eastings[0];
    double t_y1_fire = (ny_fire / 2. + sry) * dy_fire + lat2northings[0];

    std::cout << "Xform: " << srx << ", " << sry
              << "; " << nx_fire << ", " << ny_fire
              << "; " << dx_fire << ", " << dy_fire
              << "; " << t_x0_fire << ", " << t_y1_fire << std::endl;
    // END
    // May not need that section above between BEGIN - END

    // nx_atm / 2. * atm_dx + e
    // double x0_fire = t_x0_fire; // -fm_nx / 2.0 * dxf + lon2eastings[0];
    // ny_atm / 2. * atm_dy + n
    // double y1_fire = t_y1_fire; // -fm_ny / 2.0 * dyf + lat2northings[0];


    // End of Conditional for processing WRF Fire Mesh Terrain data
  }

  //
  // Process Atmospheric Mesh Wind Profiles
  //    remainder of code is for pulling out wind profiles
  //

  // Need atm mesh sizes, stored in nx_atm and ny_atm
  // top grid dimension stored in BOTTOM-TOP_GRID_DIMENSION
  atm_nz = 1;
  gblAttIter = globalAttributes.find("BOTTOM-TOP_GRID_DIMENSION");
  gblAttIter->second.getValues(&atm_nz);

  std::cout << "WRF Atmospheric Mesh Domain (nx, ny, nz): " << atm_nx << " X " << atm_ny << " X " << atm_ny << " cells." << std::endl;

  std::vector<size_t> atm_startIdx = { 0, 0, 0 };
  std::vector<size_t> atm_counts = { 1,
                                     static_cast<unsigned long>(atm_ny),
                                     static_cast<unsigned long>(atm_nx) };

  std::vector<double> atm_xlong(atm_nx * atm_ny);
  std::vector<double> atm_xlat(atm_nx * atm_ny);
  wrfInputFile.getVar("XLONG").getVar(atm_startIdx, atm_counts, atm_xlong.data());
  wrfInputFile.getVar("XLAT").getVar(atm_startIdx, atm_counts, atm_xlat.data());

  // Note
  // This code does not implement the new domain borders option that
  // Matthieu put in the Matlab code
  //

  // Wind data vertical positions, stored in PHB and PH
  //
  // For example, 360 time series, 41 vertical, 114 x 114
  // SUBDATASET_12_DESC=[360x41x114x114] PH (32-bit floating-point)
  // SUBDATASET_13_NAME=NETCDF:"/scratch/Downloads/RXCwrfout_d07_2012-11-11_15-21":PHB
  //
  // SUBDATASET_13_DESC=[360x41x114x114] PHB (32-bit floating-point)
  // SUBDATASET_14_NAME=NETCDF:"/scratch/Downloads/RXCwrfout_d07_2012-11-11_15-21":T
  //
  std::vector<size_t> atmStartIdx = { 0, 0, 0, 0 };
  std::vector<size_t> atmCounts = { 1,
                                    static_cast<unsigned long>(atm_nz),
                                    static_cast<unsigned long>(atm_ny),
                                    static_cast<unsigned long>(atm_nx) };

  std::vector<double> phbData(2 * atm_nz * atm_ny * atm_nx);
  wrfInputFile.getVar("PHB").getVar(atmStartIdx, atmCounts, phbData.data());

  std::vector<double> phData(2 * atm_nz * atm_ny * atm_nx);
  wrfInputFile.getVar("PH").getVar(atmStartIdx, atmCounts, phData.data());

  std::cout << "Reading PHB and PH from Atmospheric Mesh..." << std::endl;

  //
  // Calculate the height  (PHB + PH) / 9.81
  //
  std::vector<double> heightData(2 * atm_nz * atm_ny * atm_nx);
  for (int t = 0; t < 2; t++) {
    for (int k = 0; k < atm_nz; k++) {
      for (int j = 0; j < atm_ny; j++) {
        for (int i = 0; i < atm_nx; i++) {
          int l_idx = t * (atm_nz * atm_ny * atm_nx) + k * (atm_ny * atm_nx) + j * atm_nx + i;
          heightData[l_idx] = (phbData[l_idx] + phData[l_idx]) / 9.81;
        }
      }
    }
  }
  std::cout << "\tpressure-based height computed." << std::endl;

  // calculate number of altitudes to store, which is one less than
  // nz
  atmCounts[1] = atm_nz - 1;

  //
  // Wind components are on staggered grid in U and V
  //
  // When these are read in, we will read one additional cell in X
  // and Y, for U and V, respectively.

  std::cout << "Reading U and V staggered wind input from WRF..." << std::endl;

  atmCounts[3] = atm_nx + 1;
  std::vector<double> UStaggered(2 * (atm_nz - 1) * atm_ny * (atm_nx + 1));
  wrfInputFile.getVar("U").getVar(atmStartIdx, atmCounts, UStaggered.data());

  atmCounts[2] = atm_ny + 1;
  atmCounts[3] = atm_nx;
  std::vector<double> VStaggered(2 * (atm_nz - 1) * (atm_ny + 1) * atm_nx);
  wrfInputFile.getVar("V").getVar(atmStartIdx, atmCounts, VStaggered.data());
  atmCounts[2] = atm_ny;// reset to atm_ny

  // But then, U and V are on standard nx and ny atmos mesh, cell
  // centered space
  std::vector<double> U(2 * (atm_nz - 1) * atm_ny * atm_nx);
  std::vector<double> V(2 * (atm_nz - 1) * atm_ny * atm_nx);
  for (int t = 0; t < 2; t++) {
    for (int k = 0; k < (atm_nz - 1); k++) {
      for (int j = 0; j < atm_ny; j++) {
        for (int i = 0; i < atm_nx; i++) {

          int l_idx = t * ((atm_nz - 1) * atm_ny * atm_nx) + k * (atm_ny * atm_nx) + j * atm_nx + i;

          int l_xp1_idx = t * ((atm_nz - 1) * atm_ny * atm_nx) + k * (atm_ny * atm_nx) + j * atm_nx + i + 1;
          int l_yp1_idx = t * ((atm_nz - 1) * atm_ny * atm_nx) + k * (atm_ny * atm_nx) + (j + 1) * atm_nx + i;

          U[l_idx] = 0.5 * (UStaggered[l_idx] + UStaggered[l_xp1_idx]);
          V[l_idx] = 0.5 * (VStaggered[l_idx] + VStaggered[l_yp1_idx]);
        }
      }
    }
  }

  std::cout << "\twind input complete" << std::endl;

  // Calculate CoordZ
  std::vector<double> coordZ(2 * (atm_nz - 1) * atm_ny * atm_nx);
  for (int t = 0; t < 2; t++) {
    for (int k = 0; k < (atm_nz - 1); k++) {
      for (int j = 0; j < atm_ny; j++) {
        for (int i = 0; i < atm_nx; i++) {
          int l_idx = t * ((atm_nz - 1) * atm_ny * atm_nx) + k * (atm_ny * atm_nx) + j * atm_nx + i;
          int l_kp1_idx = t * ((atm_nz - 1) * atm_ny * atm_nx) + (k + 1) * (atm_ny * atm_nx) + j * atm_nx + i;
          coordZ[l_idx] = 0.5 * (heightData[l_idx] + heightData[l_kp1_idx]);
        }
      }
    }
  }


  //
  // wind speed sqrt(u*u + v*v);
  //
  std::vector<double> wsData(2 * (atm_nz - 1) * atm_ny * atm_nx);
  for (int t = 0; t < 2; t++) {
    for (int k = 0; k < (atm_nz - 1); k++) {
      for (int j = 0; j < atm_ny; j++) {
        for (int i = 0; i < atm_nx; i++) {
          int l_idx = t * ((atm_nz - 1) * atm_ny * atm_nx) + k * (atm_ny * atm_nx) + j * atm_nx + i;
          wsData[l_idx] = sqrt(U[l_idx] * U[l_idx] + V[l_idx] * V[l_idx]);
        }
      }
    }
  }

  std::cout << "Wind speed computed." << std::endl;


  //
  // compute wind direction
  //
  std::vector<double> wdData(2 * (atm_nz - 1) * atm_ny * atm_nx);
  for (int t = 0; t < 2; t++) {
    for (int k = 0; k < (atm_nz - 1); k++) {
      for (int j = 0; j < atm_ny; j++) {
        for (int i = 0; i < atm_nx; i++) {
          int l_idx = t * ((atm_nz - 1) * atm_ny * atm_nx) + k * (atm_ny * atm_nx) + j * atm_nx + i;

          if (U[l_idx] > 0.0)
            wdData[l_idx] = 270.0 - (180.0 / c_PI) * atan(V[l_idx] / U[l_idx]);
          else
            wdData[l_idx] = 90.0 - (180.0 / c_PI) * atan(V[l_idx] / U[l_idx]);
        }
      }
    }
  }

  std::cout << "Wind direction computed." << std::endl;

  // use the Fz0 rather than LU_INDEX

  // read LU -- depends on if reading the restart or the output file
  // roughness length...
  // LU = ncread(WRFFile,'LU_INDEX');
  // SimData.LU = LU(SimData.XSTART:SimData.XEND, SimData.YSTART:SimData.YEND,1)';
  std::cout << "Reading LU_INDEX and computing roughness lengths..." << std::endl;

  std::vector<size_t> atmStartIdx_2D = { 0, 0, 0 };
  std::vector<size_t> atmCounts_2D = { 1,
                                       static_cast<unsigned long>(atm_ny),
                                       static_cast<unsigned long>(atm_nx) };

  std::vector<float> luData(atm_ny * atm_nx);
  wrfInputFile.getVar("LU_INDEX").getVar(atmStartIdx_2D, atmCounts_2D, luData.data());

  // Computes a roughness length array covering each point of the grid
  // % In case 1 INPUT file must be WRF RESTART file
  // % In case 2 INPUT file must be WRF OUTPUT
  // % In case 3 INPUT variable is a constant value
  std::vector<float> z0Data(atm_ny * atm_nx, 0.0);
  if (m_Z0Flag == 1) {
    // supposed to come from the WRF restart file in the "Z0" field
    wrfInputFile.getVar("Z0").getVar(atmStartIdx_2D, atmCounts_2D, z0Data.data());
  } else if (m_Z0Flag == 2) {

    for (int j = 0; j < atm_ny; j++) {
      for (int i = 0; i < atm_nx; i++) {
        int l_idx = j * atm_nx + i;

        z0Data[l_idx] = lookupLandUse(luData[l_idx]);
      }
    }

  } else if (m_Z0Flag == 3) {
    // supposed to be some other data source
    std::fill(z0Data.begin(), z0Data.end(), 0.1);
  } else {
    std::cerr << "Unknown Z0Flag..." << std::endl;
    assert(m_Z0Flag > 0 && m_Z0Flag < 4);
  }


  // /////////////////////////////////////////////////////////
  //
  // This section of the code reads in the wind speeds and direction
  // at different ATM mesh sensors to create the input wind profiles
  //
  // /////////////////////////////////////////////////////////

  std::cout << "Reading and processing WRF Stations for input wind profiles..." << std::endl;

  std::cout << "Dim Count: " << wrfInputFile.getVar("U0_FMW").getDimCount() << std::endl;
  for (auto dIdx = 0u; dIdx < wrfInputFile.getVar("U0_FMW").getDimCount(); dIdx++) {
    NcDim dim = wrfInputFile.getVar("U0_FMW").getDim(dIdx);
    std::cout << "Dim " << dIdx << ", Size=" << dim.getSize() << std::endl;
  }

  // Extract time dim size
  NcDim dim = wrfInputFile.getVar("U0_FMW").getDim(0);
  int timeSize = dim.getSize();
  std::cout << "Number of time series:  " << timeSize << std::endl;

  // Extract height dim size
  dim = wrfInputFile.getVar("U0_FMW").getDim(1);
  int hgtSize = dim.getSize();

  std::vector<size_t> interpWinds_StartIdx = { timeSize - 1, 0, 0, 0 };
  std::vector<size_t> interpWinds_counts = { 1,
                                             hgtSize,
                                             static_cast<unsigned long>(fm_ny),
                                             static_cast<unsigned long>(fm_nx) };

  u0_fmw.resize(1 * hgtSize * fm_ny * fm_nx);
  wrfInputFile.getVar("U0_FMW").getVar(interpWinds_StartIdx, interpWinds_counts, u0_fmw.data());

  v0_fmw.resize(1 * hgtSize * fm_ny * fm_nx);
  wrfInputFile.getVar("V0_FMW").getVar(interpWinds_StartIdx, interpWinds_counts, v0_fmw.data());

  w0_fmw.resize(1 * hgtSize * fm_ny * fm_nx);
  wrfInputFile.getVar("W0_FMW").getVar(interpWinds_StartIdx, interpWinds_counts, w0_fmw.data());

  //
  // Compute the Checksum
  //

  // read the checksum
  std::vector<size_t> chksum_StartIdx = { timeSize - 1 };
  std::vector<size_t> chksum_counts = { 1 };

  int wrfCHSUM0_FMW = 0;
  wrfInputFile.getVar("CHSUM0_FMW").getVar(chksum_StartIdx, chksum_counts, &wrfCHSUM0_FMW);
  std::cout << "WRF CHSUM0_FMW = " << wrfCHSUM0_FMW << std::endl;

  int chkSum = 0;
  chkSum = checksum(chkSum, u0_fmw);
  chkSum = checksum(chkSum, v0_fmw);
  chkSum = checksum(chkSum, w0_fmw);
  std::cout << "QES checksum of FMW fields: " << chkSum << std::endl;

  assert(wrfCHSUM0_FMW == chkSum);

  // Read the heights
  std::vector<size_t> interpWindsHT_StartIdx = { timeSize - 1, 0, 0, 0 };
  std::vector<size_t> interpWindsHT_counts = { 1,
                                               hgtSize };

  ht_fmw.resize(1 * hgtSize);
  wrfInputFile.getVar("HT_FMW").getVar(interpWindsHT_StartIdx, interpWindsHT_counts, ht_fmw.data());


  // ////////////////////////
  // sampling strategy
  int stepSize = sensorSample;

  // Only need to keep track of sensors that are WITHIN our actual
  // domain space related to the nx X ny of the QES domain.  The
  // atm_nx and atm_ny may be quite a bit larger.

  // These hard-coded values do not seem like they should exist for
  // ALL cases -- they were in Mattiheu's original code.  We need to
  // change to something per domain or calculated per domain. -Pete
  float minWRFAlt = 20;
  float maxWRFAlt = 250;

  // max altitude to pull for wind profiles should be based on fm_nz
  // and dz since that's the highest value

  if (sensorsOnly) {
    // Use nz * dz
    maxWRFAlt = 90 * 3.0;// again, only works with dz=1
  } else {
    maxWRFAlt = fm_nz;// again, only works with dz=1
  }

  std::vector<double> atm_hgt(atm_nx * atm_ny);
  wrfInputFile.getVar("HGT").getVar(atm_startIdx, atm_counts, atm_hgt.data());

  std::cout << "Max WRF Alt: " << maxWRFAlt << std::endl;

  //
  // Walk over the atm mesh, extract wind profiles for stations
  //

  std::cout << std::setprecision(9) << "UTM(LL): " << domainUTMx << ", " << domainUTMy << ", Zone=" << UTMZone << "(" << zoneUTM << ")"
            << "\n";

  std::cout << "DIM: " << dimX << ", " << dimY << "\n";

  double domainUTMx_UR = domainUTMx + dimX;
  double domainUTMy_UR = domainUTMy + dimY;

  std::cout << std::setprecision(9) << "UTM(UR): " << domainUTMx_UR << ", " << domainUTMy_UR << std::endl;

  double c_lat_ll, c_long_ll;
  double c_lat_ur, c_long_ur;

  UTMConv(c_long_ll, c_lat_ll, domainUTMx, domainUTMy, UTMZone, 1);
  UTMConv(c_long_ur, c_lat_ur, domainUTMx_UR, domainUTMy_UR, UTMZone, 1);

  std::cout << std::setprecision(9) << "LL: " << c_lat_ll << ", " << c_long_ll << std::endl;
  std::cout << std::setprecision(9) << "UR: " << c_lat_ur << ", " << c_long_ur << std::endl;

  for (int yIdx = 0; yIdx < atm_ny; yIdx += stepSize) {
    for (int xIdx = 0; xIdx < atm_nx; xIdx += stepSize) {

      stationData sd;

      int atm_idx = (yIdx * atm_nx) + xIdx;
      // std::cout << std::setprecision(9)
      // << "Lat: " << atm_xlat[ atm_idx ]
      // << ", Long: " << atm_xlong[ atm_idx ]
      // << std::endl;

      if (((atm_xlong[atm_idx] < c_long_ur) && (atm_xlong[atm_idx] > c_long_ll)) &&

          ((atm_xlat[atm_idx] < c_lat_ur) && (atm_xlat[atm_idx] > c_lat_ll))) {

        // If geoStationData is within [UTMX,UTMY] X [UTMX+(numMetersFromDEM_x), UTMY+(numMetersFromDEM_y)]
        // then
        //   convert geoStationData to local QES coord
        //   lXCoord = geo_xCoord_UTMx - UTMx + halo_x
        //   lYCoord = geo_yCoord_UTMy - UTMy + halo_y

        // convert lat long to utm
        double utmStatX;
        double utmStatY;
        int newZone = (int)floor((atm_xlong[atm_idx] + 180) / 6) + 1;

        std::cout << "\t zone = " << newZone << std::endl;
        double latStat = atm_xlat[atm_idx];
        double longStat = atm_xlong[atm_idx];
        UTMConv(longStat, latStat, utmStatX, utmStatY, newZone, 0);

        std::cout << "Stat Lat/Long: " << latStat << ", " << longStat << " and UTM " << utmStatX << ", " << utmStatY << ", Zone: " << newZone << std::endl;

        sd.xCoord = utmStatX - domainUTMx;// need halo still    xIdx * atm_dx;  // use actual position
        sd.yCoord = utmStatY - domainUTMy;// sd.yCoord = yIdx * atm_dy;  // "

        std::cout << "\tAdding SD: " << sd.xCoord << ", " << sd.yCoord << std::endl;

        // Pull Z0
        sd.z0 = z0Data[yIdx * atm_nx + xIdx];

        sd.profiles.resize(1);// 2 time series

        // Use the last time step
        for (int t = 0; t < 1; t++) {

          std::cout << "Time: " << t << std::endl;

          // At this X, Y, look through all heights and
          // accumulate the heights that exist between our min
          // and max
          for (int k = 0; k < (atm_nz - 1); k++) {

            int l_idx = t * ((atm_nz - 1) * atm_ny * atm_nx) + k * (atm_ny * atm_nx) + yIdx * atm_nx + xIdx;


            // only works for dz = 1 so will need to
            // incorporate that...
            // if (coordZ[l_idx] <= fm_nz) {
            if (coordZ[l_idx] < maxWRFAlt) {
              // if (coordZ[l_idx] >= minWRFAlt && coordZ[l_idx] <= maxWRFAlt) {

              // Use ZSF + FZ0 -- should be built into coordZ it seems

              std::cout << "profile height at " << k << ": " << coordZ[l_idx] - atm_hgt[yIdx * atm_nx + xIdx] << "\n";

              profData profEl;
              profEl.zCoord = coordZ[l_idx] - atm_hgt[yIdx * atm_nx + xIdx];
              profEl.ws = wsData[l_idx];
              profEl.wd = wdData[l_idx];

              sd.profiles[t].push_back(profEl);
            }
          }
        }

        statData.push_back(sd);
      }
    }
  }
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
  for (auto i = 0u; i < dims.size(); i++) {
    std::cout << "Dim: " << dims[i].getName() << ", ";
    if (dims[i].isUnlimited())
      std::cout << "Unlimited (" << dims[i].getSize() << ")" << std::endl;
    else
      std::cout << dims[i].getSize() << std::endl;
    totalDim *= dims[i].getSize();
  }
  std::cout << "PHB att count: " << phbVar.getAttCount() << std::endl;
  std::map<std::string, NcVarAtt> phbVar_attrMap = phbVar.getAtts();
  for (std::map<std::string, NcVarAtt>::const_iterator ci = phbVar_attrMap.begin();
       ci != phbVar_attrMap.end();
       ++ci) {
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
  double *phbData = new double[subsetDim];
  std::vector<size_t> starts = { 0, 0, 0, 0 };
  std::vector<size_t> counts = { 2, 41, 114, 114 };// depends on order of dims
  phbVar.getVar(starts, counts, phbData);

  dumpWRFDataArray("PHB Subset", phbData, 2, 41, 114, 114);

  //
  // Extraction of the Wind data vertical position
  //
  NcVar phVar = wrfInputFile.getVar("PH");

  double *phData = new double[subsetDim];
  phVar.getVar(starts, counts, phData);


  //
  /// Height
  //
  double *heightData = new double[subsetDim];
  for (auto l = 0; l < subsetDim; l++) {
    heightData[l] = (phbData[l] + phData[l]) / 9.81;
  }

  // Extraction of the Ustagg
  // Ustagg = ncread(SimData.WRFFile,'U');
  // Ustagg = Ustagg(SimData.XSTART:SimData.XEND +1, SimData.YSTART:SimData.YEND, :, SimData.TIMEVECT);
  NcVar uStaggered = wrfInputFile.getVar("U");

  std::vector<NcDim> ustagg_dims = uStaggered.getDims();
  for (auto i = 0u; i < ustagg_dims.size(); i++) {
    std::cout << "Dim: " << ustagg_dims[i].getName() << ", ";
    if (ustagg_dims[i].isUnlimited())
      std::cout << "Unlimited (" << ustagg_dims[i].getSize() << ")" << std::endl;
    else
      std::cout << ustagg_dims[i].getSize() << std::endl;
  }

  // time, Z, Y, X is order
  starts.clear();
  counts.clear();
  starts = { 0, 0, 0, 0 };
  counts = { 2, 40, 114, 115 };
  subsetDim = 1;
  for (auto i = 0u; i < counts.size(); i++) {
    subsetDim *= (counts[i] - starts[i]);
  }

  double *uStaggeredData = new double[subsetDim];
  uStaggered.getVar(starts, counts, uStaggeredData);
  dumpWRFDataArray("Ustagg", uStaggeredData, 2, 40, 114, 115);

  //
  // Vstagg = ncread(SimData.WRFFile,'V');
  // Vstagg = Vstagg(SimData.XSTART:SimData.XEND, SimData.YSTART:SimData.YEND +1, :, SimData.TIMEVECT);
  //
  NcVar vStaggered = wrfInputFile.getVar("V");

  starts.clear();
  counts.clear();
  starts = { 0, 0, 0, 0 };
  counts = { 2, 40, 115, 114 };
  subsetDim = 1;
  for (auto i = 0u; i < counts.size(); i++)
    subsetDim *= (counts[i] - starts[i]);

  double *vStaggeredData = new double[subsetDim];
  vStaggered.getVar(starts, counts, vStaggeredData);


  //
  // %% Centering values %%
  // SimData.NbAlt = size(Height,3) - 1;
  //
  int nbAlt = 40;// zDim - 1 but hack for now -- need to be computed

  // ///////////////////////////////////
  // U = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
  // for x = 1:SimData.nx
  //   U(x,:,:,:) = .5*(Ustagg(x,:,:,:) + Ustagg(x+1,:,:,:));
  // end
  // ///////////////////////////////////

  // Just make sure we've got the write dims here
  nx = 114;
  ny = 114;

  std::vector<double> U(nx * ny * nbAlt * 2, 0.0);
  for (auto t = 0; t < 2; t++) {
    for (auto z = 0; z < nbAlt; z++) {
      for (auto y = 0; y < ny; y++) {
        for (auto x = 0; x < nx; x++) {

          auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
          auto idxP1x = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + (x + 1);

          U[idx] = 0.5 * (uStaggeredData[idx] + uStaggeredData[idxP1x]);
        }
      }
    }
  }
  dumpWRFDataArray("U", U.data(), 2, 40, 114, 114);

  // V = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
  // for y = 1:SimData.ny
  //    V(:,y,:,:) = .5*(Vstagg(:,y,:,:) + Vstagg(:,y+1,:,:));
  // end
  std::vector<double> V(nx * ny * nbAlt * 2, 0.0);
  for (auto t = 0; t < 2; t++) {
    for (auto z = 0; z < nbAlt; z++) {
      for (auto y = 0; y < ny; y++) {
        for (auto x = 0; x < nx; x++) {

          auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
          auto idxP1y = t * (nbAlt * ny * nx) + z * (ny * nx) + (y + 1) * (nx) + x;

          V[idx] = 0.5 * (vStaggeredData[idx] + vStaggeredData[idxP1y]);
        }
      }
    }
  }
  dumpWRFDataArray("V", V.data(), 2, 40, 114, 114);

  // SimData.CoordZ = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
  // for k = 1:SimData.NbAlt
  //    SimData.CoordZ(:,:,k,:) = .5*(Height(:,:,k,:) + Height(:,:,k+1,:));
  // end
  std::vector<double> coordZ(nx * ny * nbAlt * 2, 0.0);
  for (auto t = 0; t < 2; t++) {
    for (auto z = 0; z < nbAlt; z++) {
      for (auto y = 0; y < ny; y++) {
        for (auto x = 0; x < nx; x++) {

          auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
          auto idxP1z = t * (nbAlt * ny * nx) + (z + 1) * (ny * nx) + y * (nx) + x;

          coordZ[idx] = 0.5 * (heightData[idx] + heightData[idxP1z]);
        }
      }
    }
  }

  // %% Velocity and direction %%
  // SimData.WS = sqrt(U.^2 + V.^2);
  std::vector<double> WS(nx * ny * nbAlt * 2, 0.0);
  for (auto t = 0; t < 2; t++) {
    for (auto z = 0; z < nbAlt; z++) {
      for (auto y = 0; y < ny; y++) {
        for (auto x = 0; x < nx; x++) {
          auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
          WS[idx] = sqrt(U[idx] * U[idx] + V[idx] * V[idx]);
        }
      }
    }
  }
  dumpWRFDataArray("WS", WS.data(), 2, nbAlt, ny, nx);


  // SimData.WD = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
  std::vector<double> WD(nx * ny * nbAlt * 2, 0.0);
  for (auto t = 0; t < 2; t++) {
    for (auto z = 0; z < nbAlt; z++) {
      for (auto y = 0; y < ny; y++) {
        for (auto x = 0; x < nx; x++) {
          auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
          if (U[idx] > 0) {
            WD[idx] = 270.0 - (180.0 / c_PI) * atan(V[idx] / U[idx]);
          } else {
            WD[idx] = 90.0 - (180.0 / c_PI) * atan(V[idx] / U[idx]);
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
  int nx2 = floor(sqrt(m_maxTerrainSize * nx / (float)ny));
  int ny2 = floor(sqrt(m_maxTerrainSize * ny / (float)nx));

  // SimData.nx = nx2;
  // SimData.ny = ny2;
  m_dx = m_dx * nx / (float)nx2;
  m_dy = m_dy * ny / (float)ny2;

  std::cout << "Resizing from " << nx << " by " << ny << " to " << nx2 << " by " << ny2 << std::endl;

  // Terrain
  // SimData.Relief = imresize(SimData.Relief,[ny2,nx2]);
  std::vector<double> reliefResize(nx2 * ny2 * nbAlt * 2, 0.0);

  //
  // Fake this for now with nearest neighbor... implement bicubic
  // later
  //

  float scaleX = nx / (float)nx2;
  float scaleY = ny / (float)ny2;

  for (auto y = 0; y < ny2; y++) {
    for (auto x = 0; x < nx2; x++) {

      // Map this into the larger vector
      int largerX = (int)floor(x * scaleX);
      int largerY = (int)floor(y * scaleY);

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
  std::cout << "[" << name << "] WRF Data Dump" << std::endl
            << "==========================" << std::endl;

  // This output is analagous to Matlab's Columns 1 through dimY style of output
  for (auto t = 0; t < dimT; t++) {
    for (auto z = 0; z < dimZ; z++) {
      std::cout << "Slice: (t=" << t << ", z=" << z << ")" << std::endl;
      for (auto x = 0; x < dimX; x++) {
        for (auto y = 0; y < dimY; y++) {

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


float WRFInput::lookupLandUse(int luIdx)
{
  switch (luIdx) {

  case 1:// %%% Evergreen needleleaf forest
    return 0.5;
  case 2:// %%% Evergreeen broadleaf forest
    return 0.5;
  case 3:// %%% Deciduous needleleaf forest
    return 0.5;
  case 4:// %%% Deciduous broadleaf forest
    return 0.5;
  case 5:// %%% Mixed forests
    return 0.5;
  case 6:// %%% Closed Shrublands
    return 0.1;
  case 7:// %%% Open Shrublands
    return 0.1;
  case 8:// %%% Woody Savannas
    return 0.15;
  case 9:// %%% Savannas
    return 0.15;
  case 10:// %%% Grasslands
    return 0.075;
  case 11:// %%% Permanent wetlands
    return 0.3;
  case 12:// %%% Croplands
    return 0.075;
  case 13:// %%% Urban and built-up land
    return 0.5;
  case 14:// %%% Cropland/natural vegetation mosaic
    return 0.065;
  case 15:// %%% Snow or ice
    return 0.01;
  case 16:// %%% Barren or sparsely vegetated
    return 0.065;
  case 17:// %%% Water
    return 0.0001;
  case 18:// %%% Wooded tundra
    return 0.15;
  case 19:// %%% Mixed tundra
    return 0.1;
  case 20:// %%% Barren tundra
    return 0.06;
  case 21:// %%% Lakes
    return 0.0001;

  default:
    return 0.1;
  }
}


void WRFInput::endWRFSession()
{
  int wrfEndFrameNum = -99;

  // Need to output the wrfFRAME0 back to the FRAME_FMW
  std::vector<size_t> chksum_StartIdx = { 0 };
  std::vector<size_t> chksum_counts = { 1 };

  NcVar field_FRAME = wrfInputFile.getVar("FRAME_FMW");
  field_FRAME.putVar(chksum_StartIdx, chksum_counts, &wrfEndFrameNum);

  // close file
  wrfInputFile.close();
}

void WRFInput::updateFromWRF()
{
  // Only perform if doing the coupling
  if (m_performWRFRunCoupling) {

    // Wait until WRF has run and placed the correct checksum and timestamp number in the file

    //
    // check for next frame info
    //
    std::cout << "Waiting for next frame: " << nextWRFFrameNum << std::endl;

    NcDim fmwdim = wrfInputFile.getVar("U0_FMW").getDim(0);
    int fmwTimeSize = fmwdim.getSize();

    std::vector<size_t> fmw_StartIdx = { fmwTimeSize - 1 };
    std::vector<size_t> fmw_counts = { 1 };

    int numWait = 0;
    int maxWait = 20;

    int wrfFRAME0_FMW = -1;
    wrfInputFile.getVar("FRAME0_FMW").getVar(fmw_StartIdx, fmw_counts, &wrfFRAME0_FMW);
    while (wrfFRAME0_FMW != nextWRFFrameNum && (numWait < maxWait)) {
      std::cout << "Waiting for FRAME0_FMW to be ====> " << nextWRFFrameNum << ", received " << wrfFRAME0_FMW << std::endl;

      // close file
      wrfInputFile.close();

      // wait a few seconds for now
      usleep(6000000);

      // re-open
      wrfInputFile.open(m_WRFFilename, NcFile::write);
      wrfInputFile.getVar("FRAME0_FMW").getVar(fmw_StartIdx, fmw_counts, &wrfFRAME0_FMW);

      numWait++;
    }

    if ((numWait == maxWait) || (wrfFRAME0_FMW < 0))
      exit(EXIT_FAILURE);

    std::cout << "WRF-QES Coupling: Frame = " << wrfFRAME0_FMW << std::endl;

    // set the current WRF FRAME0 Num
    currWRFFRAME0Num = wrfFRAME0_FMW;

    std::cout << "Reading and processing WRF Stations for input wind profiles..." << std::endl;

    std::cout << "Dim Count: " << wrfInputFile.getVar("U0_FMW").getDimCount() << std::endl;
    for (auto dIdx = 0u; dIdx < wrfInputFile.getVar("U0_FMW").getDimCount(); dIdx++) {
      NcDim dim = wrfInputFile.getVar("U0_FMW").getDim(dIdx);
      std::cout << "Dim " << dIdx << ", Size=" << dim.getSize() << std::endl;
    }

    // Extract time dim size
    NcDim dim = wrfInputFile.getVar("U0_FMW").getDim(0);
    int timeSize = dim.getSize();
    std::cout << "Number of time series:  " << timeSize << std::endl;

    // Extract height dim size
    dim = wrfInputFile.getVar("U0_FMW").getDim(1);
    int hgtSize = dim.getSize();

    std::vector<size_t> interpWinds_StartIdx = { timeSize - 1, 0, 0, 0 };
    std::vector<size_t> interpWinds_counts = { 1,
                                               hgtSize,
                                               static_cast<unsigned long>(fm_ny),
                                               static_cast<unsigned long>(fm_nx) };

    // u0_fmw.resize( 1 * hgtSize * fm_ny * fm_nx );
    wrfInputFile.getVar("U0_FMW").getVar(interpWinds_StartIdx, interpWinds_counts, u0_fmw.data());

    // v0_fmw.resize( 1 * hgtSize * fm_ny * fm_nx );
    wrfInputFile.getVar("V0_FMW").getVar(interpWinds_StartIdx, interpWinds_counts, v0_fmw.data());

    // w0_fmw.resize( 1 * hgtSize * fm_ny * fm_nx );
    wrfInputFile.getVar("W0_FMW").getVar(interpWinds_StartIdx, interpWinds_counts, w0_fmw.data());

    //
    // Compute the Checksum
    //

    // read the checksum
    std::vector<size_t> chksum_StartIdx = { timeSize - 1 };
    std::vector<size_t> chksum_counts = { 1 };

    int wrfCHSUM0_FMW = 0;
    wrfInputFile.getVar("CHSUM0_FMW").getVar(chksum_StartIdx, chksum_counts, &wrfCHSUM0_FMW);
    std::cout << "WRF CHSUM0_FMW = " << wrfCHSUM0_FMW << std::endl;

    int chkSum = 0;
    chkSum = checksum(chkSum, u0_fmw);
    chkSum = checksum(chkSum, v0_fmw);
    chkSum = checksum(chkSum, w0_fmw);
    std::cout << "WRF file output checksum of FMW fields: " << chkSum << std::endl;

    // Read the heights
    std::vector<size_t> interpWindsHT_StartIdx = { timeSize - 1, 0, 0, 0 };
    std::vector<size_t> interpWindsHT_counts = { 1,
                                                 hgtSize };

    // ht_fmw.resize( 1 * hgtSize );
    wrfInputFile.getVar("HT_FMW").getVar(interpWindsHT_StartIdx, interpWindsHT_counts, ht_fmw.data());
  }

  std::cout << "Finalized update from WRF." << std::endl;
}


void WRFInput::extractWind(WINDSGeneralData *wgd)
{
  // WRF fire mesh and wgd domain should be identical with the
  // exception of the halo region, which only exists in QES

  // We use QES dimensions here since those are used to access the
  // wind field and at the moment, the fire mesh and the QES domain
  // should match up (as stated above).

  std::vector<size_t> startIdx = { 0, 0, 0, 0 };
  std::vector<size_t> counts = { 1,
                                 static_cast<unsigned long>(fm_ny),
                                 static_cast<unsigned long>(fm_nx) };

  std::cout << "fm_nx=" << fm_nx << ", wgd->nx=" << wgd->nx << ", haloX=" << m_haloX_DimAddition << "( " << std::endl;
  std::cout << "fm_ny=" << fm_ny << ", wgd->ny=" << wgd->ny << ", haloY=" << m_haloY_DimAddition << std::endl;

  // why?
  // assert( fm_nx != (wgd->nx - 1 - 2*m_haloX_DimAddition) );
  // assert( fm_ny != (wgd->ny - 1 - 2*m_haloY_DimAddition) );

  // Initialize the two fields to 0.0
  std::vector<float> ufOut(fm_nx * fm_ny, 0.0);
  std::vector<float> vfOut(fm_nx * fm_ny, 0.0);

  auto FWH = 1.2;

  if (FWH <= wgd->dz) {
    std::cout << "Warning: resolution in z-direction is not fine enough to define above ground cells for calculating wind" << std::endl;
    std::cout << "Try running the model with finer resolution in z-direction" << std::endl;
  }

  // For all X, Y in the fire mesh space, extract terrain height and
  // fwh to then pull the wind components from QES
  for (auto i = 0; i < fm_nx - 1; i++) {
    for (auto j = 0; j < fm_ny - 1; j++) {

      // add the halo offsets to the i and j to shift them into
      // QES's space - fire mesh and wrf data do not have halo
      // extensions
      auto iQES = i + m_haloX_DimAddition;
      auto jQES = j + m_haloY_DimAddition;

      // Use qes index
      // Gets height of the terrain for each cell
      auto qes2DIdx = jQES * (wgd->nx - 1) + iQES;
      auto tHeight = wgd->terrain[qes2DIdx];

      // find the k index value at this height in the domain,
      // need to take into account the variable dz
      int kQES;
      for (size_t k = 0; k < wgd->z.size() - 1; k++) {
        kQES = k;
        if (float(tHeight + FWH) < wgd->z[k]) {
          break;
        }
      }
      // auto kQES = (int)floor( ((tHeight + FWH)/float(wgd->dz) ));

      // fire mesh idx
      auto fireMeshIdx = j * fm_nx + i;

      // 3D QES Idx
      auto qes3DIdx = kQES * (wgd->nx) * (wgd->ny) + jQES * (wgd->nx) + iQES;

      // provide cell centered data to WRF
      // -- Need to switch to linear interpolation within the cells...
      ufOut[fireMeshIdx] = 0.5 * (wgd->u[qes3DIdx + 1] + wgd->u[qes3DIdx]);
      vfOut[fireMeshIdx] = 0.5 * (wgd->v[qes3DIdx + wgd->nx] + wgd->v[qes3DIdx]);
    }
  }

  // Compute the CHSUM
  int ufvf_chsum = 0;
  ufvf_chsum = checksum(ufvf_chsum, ufOut);
  ufvf_chsum = checksum(ufvf_chsum, vfOut);
  std::cout << "WRF UF/VF output checksum: " << ufvf_chsum << std::endl;

  NcVar field_UF = wrfInputFile.getVar("UF");
  NcVar field_VF = wrfInputFile.getVar("VF");

  NcVar field_CHSUM = wrfInputFile.getVar("CHSUM_FMW");
  NcVar field_FRAME = wrfInputFile.getVar("FRAME_FMW");

  field_UF.putVar(startIdx, counts, ufOut.data());//, startIdx, counts );
  field_VF.putVar(startIdx, counts, vfOut.data());//, startIdx, counts );

  std::vector<size_t> chksum_StartIdx = { 0 };
  std::vector<size_t> chksum_counts = { 1 };

  field_CHSUM.putVar(chksum_StartIdx, chksum_counts, &ufvf_chsum);

  // Need to output the wrfFRAME0 back to the FRAME_FMW
  std::cout << "Writing last read frame num back: " << currWRFFRAME0Num << std::endl;
  field_FRAME.putVar(chksum_StartIdx, chksum_counts, &currWRFFRAME0Num);

  std::cout << "Checksum and frame updated!" << std::endl;

  wrfInputFile.sync();
  std::cout << "Wind field data written to WRF Output file: " << m_WRFFilename << std::endl;

  nextWRFFrameNum = currWRFFRAME0Num + 1;
  std::cout << "curr frame was " << currWRFFRAME0Num << ", now waiting for " << nextWRFFrameNum << std::endl;
}


void WRFInput::applyHalotoStationData(const float haloX, const float haloY)
{
  for (auto s : statData) {
    s.xCoord += haloX;
    s.yCoord += haloY;
  }
}


void WRFInput::dumpStationData() const
{
  std::ofstream statOut("statData.m");

  statOut << "statData = [" << std::endl;
  for (auto s : statData) {
    statOut << s.xCoord << " " << s.yCoord << " " << s.z0 << ";" << std::endl;
  }
  statOut << "];" << std::endl;

  statOut.close();
}
