/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file GIStool.cpp
 */

#include "GIStool.h"

/****************************************************************************************************************

  S p e c f e m 3 D  V e r s i o n  2 . 1
  ---------------------------------------

  Main authors: Dimitri Komatitsch and Jeroen Tromp
  Princeton University, USA and CNRS / INRIA / University of Pau (c) Princeton University / 
  California Institute of Technology and CNRS / INRIA / University of Pau
  July 2012

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  he Free Software Foundation; either version 2 of the License, or
  (at your option) aWGD->ny later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along
  with this program; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

  UTM (Universal Transverse Mercator) projection from the USGS

  convert geodetic longitude and latitude to UTM, and back
  use iway = ILONGLAT2UTM for long/lat to UTM, IUTM2LONGLAT for UTM to lat/long
  a list of UTM zones of the world is available at www.dmap.co.uk/utmworld.htm

  CAMx v2.03
  
  UTM_GEO performs UTM to geodetic (long/lat) translation, and back.
  
  This is a Fortran version of the BASIC program "Transverse Mercator
  Conversion", Copyright 1986, Norman J. Berls (Stefan Musarra, 2/94)
  Based on algorithm taken from "Map Projections Used by the USGS"
  by John P. SWGD->nyder, Geological Survey Bulletin 1532, USDI.
  
  Input/Output arguments:
    
           rlon                  Longitude (deg, negative for West)
           rlat                  Latitude (deg)
           rx                    UTM easting (m)
           ry                    UTM northing (m)
           UTM_PROJECTION_ZONE   UTM zone
           iway                  Conversion type
                                 ILONGLAT2UTM = geodetic to UTM
                                 IUTM2LONGLAT = UTM to geodetic

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
****************************************************************************************************************/

void GIStool::UTMConverter(float &rlon, float &rlat, float &rx, float &ry, int &UTM_PROJECTION_ZONE, bool N_HEMISPHERE, int iway)
{

  int ILONGLAT2UTM = 0, IUTM2LONGLAT = 1;
  const float PI = 3.141592653589793;
  const float degrad = PI / 180.0;
  const float raddeg = 180.0 / PI;
  const float semimaj = 6378206.40;
  const float semimin = 6356583.80;
  const float scfa = 0.99960;

  //To change hemisphere need to change the "north" variable:
  float north = 0.0;
  if (N_HEMISPHERE)
    //- north=0 for northern hemisphere and
    north = 0.0;
  else
    //- north=10000000 (10000km) for southern hemisphere
    north = 10000000.0;

  const float east = 500000.0;

  float e2, e4, e6, ep2, xx, yy, dlat, dlon, zone, cm, cmr, delam;
  float f1, f2, f3, f4, rm, rn, t, c, a, e1, u, rlat1, dlat1, c1, t1, rn1, r1, d;
  float rx_save, ry_save, rlon_save, rlat_save;

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
  if (iway == ILONGLAT2UTM) {
    UTM_PROJECTION_ZONE = std::ceil((rlon+180)/6);
  }
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
    f3 = 15.0 * (e4 / 256.0) + 45.0 * (e6 / 1024.0);
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
      f3 = 5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * pow(c1, 2.0) - 9.0 * ep2;
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

void GIStool::getConvergence(float &lon, float &lat, int &site_UTM_zone, float &convergence)
{

  float temp_lon;
  temp_lon = (6.0 * site_UTM_zone) - 183.0 - lon;
  convergence = atan(atan(temp_lon * M_PI / 180.0) * asin(lat * M_PI / 180.0)) * (180.0 / M_PI);
}
