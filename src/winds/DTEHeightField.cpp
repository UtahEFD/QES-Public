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
 * @file DTEHeightField.cpp
 * @brief Loads digital elevation data.
 *
 * @sa Cell
 * @sa Edge
 * @sa Triangle
 * @sa Vector3
 */

#include "DTEHeightField.h"
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
#define LIMIT 99999999.0f

DTEHeightField::DTEHeightField()
  : m_filename(""), m_rbMin(0.0)
{
  m_poDataset = 0;
}

DTEHeightField::DTEHeightField(const std::string &filename,
                               std::tuple<int, int, int> dim,
                               std::tuple<float, float, float> cellSize,
                               float UTMx,
                               float UTMy,
                               int OriginFlag,
                               float DEMDistanceX,
                               float DEMDistanceY)
  : DEMDistancex(DEMDistanceX), DEMDistancey(DEMDistanceY), originFlag(OriginFlag),
    m_filename(filename), m_rbMin(0.0),
    m_cellSize(cellSize), m_dim(dim),
    domain_UTMx(UTMx), domain_UTMy(UTMy)
{
  GDALAllRegister();

  load();
}

// Constructor for converting heightfield data to the internal
// representation for digital elevation
//
// Inputs need to provide the nx, ny, nz
DTEHeightField::DTEHeightField(const std::vector<double> &heightField,
                               std::tuple<int, int, int> dim,
                               std::tuple<float, float, float> cellSize,
                               float halo_x,
                               float halo_y)
  : m_cellSize(cellSize), m_dim(dim)
{
  Triangle *tPtr = 0;
  m_triList.clear();

  // local variables to hold triangle vertices
  // Vector3 tc0, tc1, tc2;

  // local variables to hold common variables
  // decomposition declarations are a C++17 extension
  // auto [nx, ny, nz] = m_dim;
  int nx = std::get<0>(m_dim);
  int ny = std::get<1>(m_dim);
  int nz = std::get<2>(m_dim);
  // auto [dx, dy, dz] = m_cellSize;
  float dx = std::get<0>(m_cellSize);
  float dy = std::get<1>(m_cellSize);
  float dz = std::get<2>(m_cellSize);

  std::cout << "Loading digital elevation data from height field\n";

  std::cout << "dimX = " << nx << ", dimY = " << ny << ", dimZ = " << nz << std::endl;
  std::cout << "cellSizes = (" << dx << ", " << dy << ", " << dz << ")" << std::endl;

  std::cout << "size of heightField = " << heightField.size() << std::endl;

  // eventually need the fm_dx and fm_dy so we can multiply into
  // correct dimensions

  int step = 1;// step size is interpretted incorrectly here for
    // fire meshes...

  // previously, with regular DEMs we can use the cellSize to
  // determine how we step over the terrain to create the actual
  // mesh... based on dx, dy...

  // This triangle mesh is in the dimensions of the height field
  // array... may not be in the domain space... hence when queried
  // later in the

  std::cout << "Adding halo to surrounding regions..." << halo_x << ", " << halo_y << std::endl;

  for (float j = 0; j < ny - 1; j += step) {
    for (float i = 0; i < nx - 1; i += step) {

      size_t idx = j * nx + i;
      if (idx > heightField.size() - 1) idx = heightField.size() - 1;

      // when pulling data from the height field and converting
      // to actual locations, we need to add the halo_x and
      // halo_y to all positions to shift the domain -- these
      // are in meters...

      //
      // Need to convert these to localized QES dimensions
      //
      float xPos = halo_x + (i * dx);
      float yPos = halo_y + (j * dy);

      Vector3 tv0(xPos, yPos, (float)heightField[idx]);// queryHeight( pafScanline, Xpixel,  Yline));

      idx = j * nx + (i + step);
      if (idx > heightField.size() - 1) {
        std::cout << "***************" << std::endl;
        idx = heightField.size() - 1;
      }

      xPos = halo_x + ((i + step) * dx);
      Vector3 tv1(xPos, yPos, (float)heightField[idx]);// queryHeight( pafScanline,  (int)(iXpixel + stepX ), Yline ) );

      idx = (j + step) * nx + i;
      if (idx > heightField.size() - 1) {
        std::cout << "---------------" << std::endl;
        idx = heightField.size() - 1;
      }

      xPos = halo_x + (i * dx);
      yPos = halo_y + ((j + step) * dy);
      Vector3 tv2(xPos, yPos, (float)heightField[idx]);// queryHeight( pafScanline, Xpixel, (int)(iYline + stepY) ));

      tPtr = new Triangle(tv0, tv1, tv2);
      m_triList.push_back(tPtr);

      idx = (j + step) * nx + i;
      if (idx > heightField.size() - 1) idx = heightField.size() - 1;

      xPos = halo_x + (i * dx);
      yPos = halo_y + ((j + step) * dy);
      Vector3 tv3(xPos, yPos, (float)heightField[idx]);// queryHeight( pafScanline,  Xpixel, (int)(iYline + stepY) ) );

      idx = j * nx + (i + step);
      if (idx > heightField.size() - 1) idx = heightField.size() - 1;
      xPos = halo_x + ((i + step) * dx);
      yPos = halo_y + (j * dy);
      Vector3 tv4(xPos, yPos, (float)heightField[idx]);//  queryHeight( pafScanline,  (int)(iXpixel + stepX) , Yline ) );

      idx = (j + step) * nx + (i + step);
      if (idx > heightField.size() - 1) idx = heightField.size() - 1;
      xPos = halo_x + ((i + step) * dx);
      yPos = halo_y + ((j + step) * dy);
      Vector3 tv5(xPos, yPos, (float)heightField[idx]);// queryHeight( pafScanline, (int)(iXpixel + stepX), (int)(iYline + stepY) ) );

      tPtr = new Triangle(tv3, tv4, tv5);
      m_triList.push_back(tPtr);
    }
  }

  std::cout << "... completed." << std::endl;

  // At end of loop above, all height field data will have been
  // converted to a triangle mesh, stored in m_triList.
}

#if 0
void DTEHeightField::loadImage()
{

  std::string filename = "/scratch/dem.png";

  std::cout << "loadImage: opening " << filename << std::endl;

  // From -- http://www.gdal.org/gdal_tutorial.html
  m_imageDataset = (GDALDataset *) GDALOpen( filename.c_str(), GA_ReadOnly );
  if( m_imageDataset == NULL )
    {
      std::cerr << "Couldn't open file: " << filename << std::endl;
      exit( EXIT_FAILURE );
    }

  printf( "Driver: %s/%s\n",
      m_imageDataset->GetDriver()->GetDescription(),
      m_imageDataset->GetDriver()->GetMetadataItem( GDAL_DMD_LONGNAME ) );

  printf( "Size is %dx%dx%d\n",
      m_imageDataset->GetRasterXSize(), m_imageDataset->GetRasterYSize(),
      m_imageDataset->GetRasterCount() );

  if( m_imageDataset->GetProjectionRef()  != NULL )
    printf( "Projection is `%s'\n", m_imageDataset->GetProjectionRef() );

  if( m_imageDataset->GetGeoTransform( m_imageGeoTransform ) == CE_None )
    {
      printf( "Origin = (%.6f,%.6f)\n",
          m_imageGeoTransform[0], m_imageGeoTransform[3] );

      printf( "Pixel Size = (%.6f,%.6f)\n",
          m_imageGeoTransform[1], m_imageGeoTransform[5] );
    }

  m_imageXSize = m_imageDataset->GetRasterXSize();
  m_imageYSize = m_imageDataset->GetRasterYSize();

  std::cout << "loadImage: image loaded." << std::endl;
}
#endif

void DTEHeightField::load()
{
  std::cout << "DTEHeightField loading DTE..." << std::endl;

  //
  // local variables to hold common variables related to the QES
  // domain
  //
  //decomposition declarations are a C++17 extension
  // auto [nx, ny, nz] = m_dim;
  int nx = std::get<0>(m_dim);
  int ny = std::get<1>(m_dim);
  int nz = std::get<2>(m_dim);
  // auto [dx, dy, dz] = m_cellSize;
  float dx = std::get<0>(m_cellSize);
  float dy = std::get<1>(m_cellSize);
  float dz = std::get<2>(m_cellSize);

  // From -- http://www.gdal.org/gdal_tutorial.html
  m_poDataset = (GDALDataset *)GDALOpen(m_filename.c_str(), GA_ReadOnly);
  if (m_poDataset == NULL) {
    std::cerr << "Couldn't open file: " << m_filename << std::endl;
    exit(EXIT_FAILURE);
  }

  printf("GDAL Driver: %s/%s\n",
         m_poDataset->GetDriver()->GetDescription(),
         m_poDataset->GetDriver()->GetMetadataItem(GDAL_DMD_LONGNAME));

  printf("\tRaster Size is %dx%dx%d\n",
         m_poDataset->GetRasterXSize(),
         m_poDataset->GetRasterYSize(),
         m_poDataset->GetRasterCount());

  // Attempt to get the spatial reference from this dataset -
  // which will help us convert into lat/long
  // In GDAL 3.0+ we can use
  // spatialRef = m_poDataset->GetSpatialRef, but in pre-3.0 versions,
  // this comes from GetProjectionRef
  if (m_poDataset->GetProjectionRef() != NULL)
    printf("\tProjection is `%s'\n", m_poDataset->GetProjectionRef());

  if (m_poDataset->GetGeoTransform(m_geoTransform) == CE_None) {
    printf("\tDEM Origin = (%.6f,%.6f)\n",
           m_geoTransform[0],
           m_geoTransform[3]);

    printf("\tPixel Size = (%.6f,%.6f)\n",
           m_geoTransform[1],
           m_geoTransform[5]);
    pixelSizeX = abs(m_geoTransform[1]);
    pixelSizeY = abs(m_geoTransform[5]);

    printf("These should be zero for north up: (%.6f, %.6f)\n",
           m_geoTransform[2],
           m_geoTransform[4]);
  }

  GDALRasterBand *poBand;
  int nBlockXSize, nBlockYSize;
  int bGotMin, bGotMax;
  // double          adfMinMax[2];

  poBand = m_poDataset->GetRasterBand(1);
  poBand->GetBlockSize(&nBlockXSize, &nBlockYSize);
  printf("\tRaster Block=%dx%d Type=%s, ColorInterp=%s\n",
         nBlockXSize,
         nBlockYSize,
         GDALGetDataTypeName(poBand->GetRasterDataType()),
         GDALGetColorInterpretationName(poBand->GetColorInterpretation()));

  adfMinMax[0] = poBand->GetMinimum(&bGotMin);
  adfMinMax[1] = poBand->GetMaximum(&bGotMax);
  if (!(bGotMin && bGotMax))
    GDALComputeRasterMinMax((GDALRasterBandH)poBand, TRUE, adfMinMax);

  printf("\tRaster Min=%.3fd and Max=%.3f\n", adfMinMax[0], adfMinMax[1]);

  if (poBand->GetOverviewCount() > 0)
    printf("\tBand has %d overviews.\n", poBand->GetOverviewCount());

  if (poBand->GetColorTable() != NULL)
    printf("\tBand has a color table with %d entries.\n",
           poBand->GetColorTable()->GetColorEntryCount());

  m_rbScale = poBand->GetScale();
  printf("Band has scale: %.4f\n", m_rbScale);

  m_rbOffset = poBand->GetOffset();
  printf("Band has offset: %.4f\n", m_rbOffset);

  m_rbNoData = poBand->GetNoDataValue();
  printf("Band has NoData value: %.4f\n", m_rbNoData);

  m_nXSize = poBand->GetXSize();
  m_nYSize = poBand->GetYSize();

  origin_x = m_geoTransform[0];
  origin_y = m_geoTransform[3] - pixelSizeY * m_nYSize;

  printf("\tDEM size: %dx%d\n", m_nXSize, m_nYSize);
  printf("\tDEM size: %.1fx%.1f\n", m_nXSize * pixelSizeX, m_nYSize * pixelSizeY);

  printf("\tDomain Origin = (%.6f,%.6f)\n", origin_x, origin_y);

  // UTMx will be correct, but need to subtract halo
  //      demMinX == UTMx - halo_x
  // UTMy needs to have domain "Y" amount added to it first
  //      demMinY = (UTMy + m_nYSize * perPixelDim) - halo_y
  //
  // then, use demMinX for transform[0] and
  //           demMinY for transform[1]
  //
  // if (we have utm, override the transofmr[0] and [3] parts...

  std::cout << "Mapping between raster coordinates and geo-referenced coordinates" << std::endl;
  double xGeo(0.0), yGeo(0.0);
  convertRasterToGeo(0, 0, xGeo, yGeo);
  printf("Raster Coordinate (0, 0):\t(%12.7f, %12.7f)\n", xGeo, yGeo);

  convertRasterToGeo(m_nXSize, 0, xGeo, yGeo);
  printf("Raster Coordinate (%d, 0):\t(%12.7f, %12.7f)\n", m_nXSize, xGeo, yGeo);

  convertRasterToGeo(m_nXSize, m_nYSize, xGeo, yGeo);
  printf("Raster Coordinate (%d, %d):\t(%12.7f, %12.7f)\n", m_nXSize, m_nYSize, xGeo, yGeo);

  convertRasterToGeo(0, m_nYSize, xGeo, yGeo);
  printf("Raster Coordinate (0, %d):\t(%12.7f, %12.7f)\n", m_nYSize, xGeo, yGeo);

  if (originFlag == 0) {
    float domain_end_x = origin_x + DEMDistancex + nx * dx;
    float domain_end_y = origin_y + DEMDistancey + ny * dy;
    float dem_end_x = origin_x + m_nXSize * pixelSizeX;
    float dem_end_y = origin_y + m_nYSize * pixelSizeY;

    if (((DEMDistancex > 0.0) || (DEMDistancey > 0.0))
        && (DEMDistancex < m_nXSize * pixelSizeX)
        && (DEMDistancex < m_nYSize * pixelSizeY)) {
      shift_x = DEMDistancex / pixelSizeX;
      shift_y = DEMDistancey / pixelSizeY;
    }

    if (((domain_end_x < dem_end_x) || (domain_end_y < dem_end_y))) {
      end_x = (dem_end_x - domain_end_x) / pixelSizeX;
      end_y = (dem_end_y - domain_end_y) / pixelSizeY;
    }

    m_nXSize = m_nXSize - shift_x - end_x;
    m_nYSize = m_nYSize - shift_y - end_y;
  } else if (originFlag == 1) {
    float domain_end_x = domain_UTMx + nx * dx;
    float domain_end_y = domain_UTMy + ny * dy;
    float dem_end_x = origin_x + m_nXSize * pixelSizeX;
    float dem_end_y = origin_y + m_nYSize * pixelSizeY;

    if (((domain_UTMx > origin_x) || (domain_UTMy > origin_y))
        && (domain_UTMx < dem_end_x)
        && (domain_UTMy < dem_end_y)) {
      shift_x = (domain_UTMx - origin_x) / pixelSizeX;
      shift_y = (domain_UTMy - origin_y) / pixelSizeY;
    }

    if (((domain_end_x < dem_end_x) || (domain_end_y < dem_end_y))
        && (domain_UTMx >= origin_x)
        && (domain_UTMy >= origin_y)) {
      end_x = (dem_end_x - domain_end_x) / pixelSizeX;
      end_y = (dem_end_y - domain_end_y) / pixelSizeY;
    }

    m_nXSize = m_nXSize - shift_x - end_x;
    m_nYSize = m_nYSize - shift_y - end_y;
  }

  pafScanline = (float *)CPLMalloc(sizeof(float) * (m_nXSize) * (m_nYSize));

  //
  // CPLErr - CE_Failure if the access fails, otherwise CE_None.
  CPLErr rasterErr = poBand->RasterIO(GF_Read, shift_x, end_y, m_nXSize, m_nYSize, pafScanline, m_nXSize, m_nYSize, GDT_Float32, 0, 0);
  if (rasterErr == CE_Failure) {
    std::cerr << "CPL RasterIO failure during DEM loading. Exiting." << std::endl;
    exit(EXIT_FAILURE);
  }


  Triangle *tPtr = 0;
  m_triList.clear();

  // double xGeo, yGeo;
  Vector3 tc0, tc1, tc2;


  std::cout << "DEM Loading\n";

  float stepX = dx / pixelSizeX;// tie back to dx, dy here.... with scaling of pixelsize
  float stepY = dy / pixelSizeY;

  assert(stepX > 0 && stepY > 0);

  for (float iYline = 0; iYline < m_nYSize - 1; iYline += stepY) {
    for (float iXpixel = 0; iXpixel < m_nXSize - 1; iXpixel += stepX) {

      int Yline = (int)iYline;
      int Xpixel = (int)iXpixel;


      // For these purposes, pixel refers to the "X" coordinate, while
      // line refers to the "Z" coordinate

      // turn localized coordinates (Yline and Xpixel) into geo-referenced values.
      // then use the geo referenced coordinate to lookup the height.


      //      xGeo = m_geoTransform[0] + Xpixel * m_geoTransform[1];
      //      yGeo = m_geoTransform[3] + Yline * m_geoTransform[5];

      // height set to y coord, change to z

      // These "should" be real unit-based triangles.. hopefully meters..
      Vector3 tv0(iXpixel * pixelSizeX, iYline * pixelSizeY, queryHeight(pafScanline, Xpixel, Yline));
      Vector3 tv1((iXpixel + stepX) * pixelSizeX, iYline * pixelSizeY, queryHeight(pafScanline, (int)(iXpixel + stepX), Yline));
      Vector3 tv2(iXpixel * pixelSizeX, (iYline + stepY) * pixelSizeY, queryHeight(pafScanline, Xpixel, (int)(iYline + stepY)));

      tPtr = new Triangle(tv0, tv1, tv2);
      m_triList.push_back(tPtr);


      Vector3 tv3(iXpixel * pixelSizeX, (iYline + stepY) * pixelSizeY, queryHeight(pafScanline, Xpixel, (int)(iYline + stepY)));

      Vector3 tv4((iXpixel + stepX) * pixelSizeX, iYline * pixelSizeY, queryHeight(pafScanline, (int)(iXpixel + stepX), Yline));

      Vector3 tv5((iXpixel + stepX) * pixelSizeX, (iYline + stepY) * pixelSizeY, queryHeight(pafScanline, (int)(iXpixel + stepX), (int)(iYline + stepY)));

      tPtr = new Triangle(tv3, tv4, tv5);
      m_triList.push_back(tPtr);
    }
    printProgress((float)(iYline + 1) / (float)(m_nYSize - 1));
  }
  std::cout << std::endl;

  // At end of loop above, all height field data will have been
  // converted to a triangle mesh, stored in m_triList.
}

DTEHeightField::~DTEHeightField()
{
  if (m_poDataset)
    GDALClose(m_poDataset);
}

void DTEHeightField::setDomain(Vector3Int &domain, Vector3 &grid)
{
  for (int i = 0; i < 3; i++) {
    min[i] = LIMIT;
    max[i] = 0.0f;
  }

  auto start = std::chrono::high_resolution_clock::now();// Start
    // recording
    // execution
    // time

  std::cout << "Setting Terrain Boundaries\n";
  for (int q = 0; q < 3; q++) {
    int triListSize = m_triList.size();

#pragma acc parallel loop
    for (int i = 0; i < triListSize; i++) {
      if (m_triList[i]->a[q] >= 0 && m_triList[i]->a[q] < min[q])
        min[q] = m_triList[i]->a[q];
      if (m_triList[i]->b[q] >= 0 && m_triList[i]->b[q] < min[q])
        min[q] = m_triList[i]->b[q];
      if (m_triList[i]->c[q] >= 0 && m_triList[i]->c[q] < min[q])
        min[q] = m_triList[i]->c[q];

      if (m_triList[i]->a[q] > max[q] && m_triList[i]->a[q] < LIMIT)
        max[q] = m_triList[i]->a[q];
      if (m_triList[i]->b[q] > max[q] && m_triList[i]->b[q] < LIMIT)
        max[q] = m_triList[i]->b[q];
      if (m_triList[i]->c[q] > max[q] && m_triList[i]->c[q] < LIMIT)
        max[q] = m_triList[i]->c[q];
    }

#pragma acc parallel loop
    for (int i = 0; i < triListSize; i++) {
      m_triList[i]->a[q] -= min[q];
      m_triList[i]->b[q] -= min[q];
      m_triList[i]->c[q] -= min[q];
    }
  }

  auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time

  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "\telapsed time: " << elapsed.count() << " s\n";// Print out elapsed execution time

  printf("Newly calculated domain size: %d %d %d\n", domain[0], domain[1], domain[2]);
}

void DTEHeightField::outputOBJ(std::string s)
{
  std::ofstream file;
  file.open(s.c_str());

  std::vector<Vector3 *> verts;
  std::vector<Vector3 *> tris;

  for (size_t i = 0; i < m_triList.size(); i++) {

    Triangle *t = m_triList[i];

    Vector3 tVs = Vector3(-1, -1, -1);

    for (size_t j = 0; j < verts.size(); j++) {
      if (t->a == (*verts[j]))
        tVs[0] = j + 1;
      if (t->b == (*verts[j]))
        tVs[1] = j + 1;
      if (t->c == (*verts[j]))
        tVs[2] = j + 1;
    }

    if (tVs[0] == -1) {
      verts.push_back(&t->a);
      tVs[0] = verts.size();
    }

    if (tVs[1] == -1) {
      verts.push_back(&t->b);
      tVs[1] = verts.size();
    }

    if (tVs[2] == -1) {
      verts.push_back(&t->c);
      tVs[2] = verts.size();
    }

    tris.push_back(&tVs);
    printProgress(((float)(i + 1) / (float)m_triList.size()) * (9.0f / 10.0f));
  }

  for (size_t i = 0; i < verts.size(); i++) {
    file << "v " << (*verts[i])[0] << " " << (*verts[i])[1] << " " << (*verts[i])[2] << "\n";
    printProgress(((float)(i + 1) / (float)m_triList.size()) / 20.0f + 0.9f);
  }

  for (size_t i = 0; i < tris.size(); i++) {
    file << "f " << (*tris[i])[0] << " " << (*tris[i])[1] << " " << (*tris[i])[2] << "\n";
    printProgress(((float)(i + 1) / (float)m_triList.size()) / 20.0f + 0.95f);
  }

  file.close();
  printf("\n");
}

void DTEHeightField::printProgress(float percentage)
{
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}


#define CELL(i, j, k) ((i) + (j) * (nx - 1) + (k) * (nx - 1) * (ny - 1))
#define CLAMP(low, high, x) ((x) < (low) ? (low) : ((x) > (high) ? (high) : (x)))

void DTEHeightField::setCells(WINDSGeneralData *WGD, const WINDSInputData *WID)
{

  printf("Setting Cell Data...\n");

  std::vector<int> cutCells;

  int ii = WID->simParams->halo_x / WGD->dx;
  int jj = WID->simParams->halo_y / WGD->dy;
  int i_domain_end = ii + (m_nXSize * pixelSizeX) / WGD->dx;
  int j_domain_end = jj + (m_nYSize * pixelSizeY) / WGD->dy;

  for (int i = 0; i < WGD->nx - 2; i++)
    for (int j = 0; j < WGD->ny - 2; j++) {

      // all work here is done for each column of cells in the z direction from the xy plane.

      Vector3 corners[4];// stored from top Left in clockwise order
      if (i >= ii && j >= jj && i <= i_domain_end && j <= j_domain_end) {
        corners[0] = Vector3(i * WGD->dx,
                             j * WGD->dy,
                             CLAMP(0,
                                   max[2],
                                   queryHeight(pafScanline, ((i - ii) * WGD->dx) / pixelSizeX, ((j - jj) * WGD->dy) / pixelSizeY) - min[2]));
        corners[1] = Vector3(i * WGD->dx,
                             (j + 1) * WGD->dy,
                             CLAMP(0,
                                   max[2],
                                   queryHeight(pafScanline, ((i - ii) * WGD->dx) / pixelSizeX, (((j - jj) + 1) * WGD->dy) / pixelSizeY) - min[2]));
        corners[2] = Vector3((i + 1) * WGD->dx,
                             (j + 1) * WGD->dy,
                             CLAMP(0,
                                   max[2],
                                   queryHeight(pafScanline, (((i - ii) + 1) * WGD->dx) / pixelSizeX, (((j - jj) + 1) * WGD->dy) / pixelSizeY) - min[2]));
        corners[3] = Vector3((i + 1) * WGD->dx,
                             j * WGD->dy,
                             CLAMP(0,
                                   max[2],
                                   queryHeight(pafScanline, (((i - ii) + 1) * WGD->dx) / pixelSizeX, ((j - jj) * WGD->dy) / pixelSizeY) - min[2]));
      } else {
        corners[0] = Vector3(i * WGD->dx, j * WGD->dy, 0.0f);
        corners[1] = Vector3(i * WGD->dx, (j + 1) * WGD->dy, 0.0f);
        corners[2] = Vector3((i + 1) * WGD->dx, (j + 1) * WGD->dy, 0.0f);
        corners[3] = Vector3((i + 1) * WGD->dx, j * WGD->dy, 0.0f);
      }

      setCellPoints(i, j, WGD->nx, WGD->ny, WGD->nz, WGD->dz_array, WGD->z_face, corners, cutCells, WGD);
    }
}

void DTEHeightField::setCellPoints(int i, int j, int nx, int ny, int nz, std::vector<float> &dz_array, std::vector<float> z_face, Vector3 corners[], std::vector<int> &cutCells, WINDSGeneralData *WGD)
{
  float coordsMin, coordsMax;
  coordsMin = coordsMax = corners[0][2];
  for (int l = 1; l <= 3; l++) {
    if (coordsMin > corners[l][2]) {
      coordsMin = corners[l][2];
    } else if (coordsMax < corners[l][2]) {
      coordsMax = corners[l][2];
    }
  }

  // #pragma acc parallel loop
  for (int k = 1; k < nz - 1; k++) {
    float cellBot = z_face[k];
    float cellTop = cellBot + dz_array[k];

    if (cellTop <= coordsMin)
      WGD->icellflag[CELL(i, j, k)] = 2;
    else if (cellBot >= coordsMax)
      WGD->icellflag[CELL(i, j, k)] = 1;
    else {
      WGD->icellflag[CELL(i, j, k)] = 8;

      int cutcell_index = CELL(i, j, k);

      std::vector<Vector3> pointsInCell;
      std::vector<Edge<int>> edgesInCell;


      // Check to see the positions of the corners, the corners are always in
      // the cell, no matter what. If they exist out of bounds of the cell in the
      // Z dimension, we add them at the floor or ceiling of the cell. We can use
      // this to identify where the actual geometry of the terrain crosses the cell,
      // as anything below the mesh of points and edges is terrain, and above is air.

      int cornerPos[4] = { 0, 0, 0, 0 };// 0 is in, 1 is above, -1 is below
      // check if corners are in
      for (int l = 0; l < 4; l++)
        if (corners[l][2] >= cellBot && corners[l][2] <= cellTop) {
          pointsInCell.push_back(corners[l]);
          cornerPos[l] = 0;
        } else if (corners[l][2] < cellBot) {
          cornerPos[l] = -1;
          pointsInCell.push_back(Vector3(corners[l][0], corners[l][1], cellBot));
        } else {
          cornerPos[l] = 1;
          pointsInCell.push_back(Vector3(corners[l][0], corners[l][1], cellTop));
        }
      for (int first = 0; first < 3; first++)
        for (int second = first + 1; second < 4; second++)
          if ((first != 1 || second != 3) && (first != 0 || second != 2))
            if (cornerPos[first] == cornerPos[second])
              edgesInCell.push_back(Edge<int>(first, second));

      // check intermediates 0-1 0-2 0-3 1-2 2-3
      int intermed[4][4][2];// first two array cells are for identifying the corners, last one is top and bottom of cell
      // note, only care about about the above pairs, in 3rd index 0 is bot 1 is top

      // initialize all identifiers to -1, this is a position in the list of points that doesn't exist
      for (int di = 0; di < 4; di++)
        for (int dj = 0; dj < 4; dj++)
          intermed[di][dj][0] = intermed[di][dj][1] = -1;

      // for all considered pairs 0-1 0-2 0-3 1-2 2-3, we check to see if they span a Z-dimension boundary
      // of the cell. If they do, we add an intermediate point that stops at the cell boundaries. And we
      // update the intermediate matrix so that we know what is the index of the intermediate point.
      for (int first = 0; first < 3; first++)
        for (int second = first + 1; second < 4; second++)
          if (first != 1 || second != 3) {
            if (cornerPos[first] == 0) {
              if (cornerPos[second] < 0) {
                intermed[first][second][0] = pointsInCell.size();
                pointsInCell.push_back(Vector3(getIntermediate(corners[first], corners[second], cellBot)));
                edgesInCell.push_back(Edge<int>(second, pointsInCell.size() - 1));
                edgesInCell.push_back(Edge<int>(first, pointsInCell.size() - 1));
              } else if (cornerPos[second] > 0) {
                intermed[first][second][1] = pointsInCell.size();
                pointsInCell.push_back(Vector3(getIntermediate(corners[first], corners[second], cellTop)));
                edgesInCell.push_back(Edge<int>(second, pointsInCell.size() - 1));
                edgesInCell.push_back(Edge<int>(first, pointsInCell.size() - 1));
              }
            } else if (cornerPos[first] > 0) {

              if (cornerPos[second] == 0) {
                intermed[first][second][1] = pointsInCell.size();
                pointsInCell.push_back(Vector3(getIntermediate(corners[first], corners[second], cellTop)));
                edgesInCell.push_back(Edge<int>(first, pointsInCell.size() - 1));
                edgesInCell.push_back(Edge<int>(second, pointsInCell.size() - 1));
              } else if (cornerPos[second] < 0) {
                intermed[first][second][1] = pointsInCell.size();
                intermed[first][second][0] = pointsInCell.size() + 1;
                pointsInCell.push_back(Vector3(getIntermediate(corners[first], corners[second], cellTop)));
                pointsInCell.push_back(Vector3(getIntermediate(corners[first], corners[second], cellBot)));
                edgesInCell.push_back(Edge<int>(first, pointsInCell.size() - 2));
                edgesInCell.push_back(Edge<int>(second, pointsInCell.size() - 1));
              }
            } else {
              if (cornerPos[second] == 0) {
                intermed[first][second][0] = pointsInCell.size();
                pointsInCell.push_back(Vector3(getIntermediate(corners[first], corners[second], cellBot)));
                edgesInCell.push_back(Edge<int>(first, pointsInCell.size() - 1));
                edgesInCell.push_back(Edge<int>(second, pointsInCell.size() - 1));
              } else if (cornerPos[second] > 0) {
                intermed[first][second][1] = pointsInCell.size();
                intermed[first][second][0] = pointsInCell.size() + 1;
                pointsInCell.push_back(Vector3(getIntermediate(corners[first], corners[second], cellTop)));
                pointsInCell.push_back(Vector3(getIntermediate(corners[first], corners[second], cellBot)));
                edgesInCell.push_back(Edge<int>(second, pointsInCell.size() - 2));
                edgesInCell.push_back(Edge<int>(first, pointsInCell.size() - 1));
              }
            }
          }

      // if there is a top and bottom on any intermediate set, create an edge
      for (int first = 0; first < 3; first++)
        for (int second = first + 1; second < 4; second++)
          if (first != 1 || second != 3)
            if (intermed[first][second][0] != -1 && intermed[first][second][1] != -1)
              edgesInCell.push_back(Edge<int>(intermed[first][second][0], intermed[first][second][1]));

      // intermediates who share a corner on the same plane as them form an edge
      // unless the diagonal is also there
      for (int tier = 0; tier < 2; tier++) {
        if (cornerPos[1] == -1 + (2 * tier) && intermed[0][1][tier] != -1 && intermed[1][2][tier] != -1)//-1 for bottom, 1 for top
        {
          edgesInCell.push_back(Edge<int>(intermed[0][1][tier], intermed[1][2][tier]));
        }
        if (cornerPos[3] == -1 + (2 * tier) && intermed[0][3][tier] != -1 && intermed[2][3][tier] != -1)//-1 for bottom, 1 for top
        {
          edgesInCell.push_back(Edge<int>(intermed[0][3][tier], intermed[2][3][tier]));
        }
      }


      // if the diagonal is completely in the cell create a mid a
      // and attatch to all intermediates or corners if the intermeds doesn't exist
      if (cornerPos[0] == 0 && cornerPos[2] == 0) {
        pointsInCell.push_back(Vector3((corners[0][0] + corners[2][0]) / 2.0f,
                                       (corners[0][1] + corners[2][1]) / 2.0f,
                                       (corners[0][2] + corners[2][2]) / 2.0f));
        int newP = pointsInCell.size() - 1;
        edgesInCell.push_back(Edge<int>(0, newP));
        edgesInCell.push_back(Edge<int>(2, newP));
        if (cornerPos[1] == 0)
          edgesInCell.push_back(Edge<int>(1, newP));
        else
          for (int tier = 0; tier < 2; tier++) {
            if (intermed[0][1][tier] != -1)
              edgesInCell.push_back(Edge<int>(intermed[0][1][tier], newP));
            if (intermed[1][2][tier] != -1)
              edgesInCell.push_back(Edge<int>(intermed[1][2][tier], newP));
          }
        if (cornerPos[3] == 0)
          edgesInCell.push_back(Edge<int>(3, newP));
        else
          for (int tier = 0; tier < 2; tier++) {
            if (intermed[0][3][tier] != -1)
              edgesInCell.push_back(Edge<int>(intermed[0][3][tier], newP));
            if (intermed[2][3][tier] != -1)
              edgesInCell.push_back(Edge<int>(intermed[2][3][tier], newP));
          }
      }
      // if there is one diagonal intermed, run the same idea as above
      // note: code will be quite different. But this should connect
      // to all intermediates, and all corners if they are missing an
      // intermediate pair.---- != is essentially XOR
      else if ((intermed[0][2][0] != -1) != (intermed[0][2][1] != -1)) {
        int midP = (intermed[0][2][0] != -1 ? intermed[0][2][0] : intermed[0][2][1]);
        //only need to check 1 and 3 corners
        //since there is only one intermediate on the diagonal, either 0 or 2 exists in the cell
        //because of this, if 1 or 3 exists in the cell the intermediate always connects to it
        if ((cornerPos[1] == -1 && (intermed[0][1][0] == -1 || intermed[1][2][0] == -1))
            || (cornerPos[1] == 1 && (intermed[0][1][1] == -1 || intermed[1][2][1] == -1))
            || cornerPos[1] == 0)
          edgesInCell.push_back(Edge<int>(1, midP));

        if ((cornerPos[3] == -1 && (intermed[0][3][0] == -1 || intermed[2][3][0] == -1))
            || (cornerPos[3] == 1 && (intermed[0][3][1] == -1 || intermed[2][3][1] == -1))
            || cornerPos[3] == 0)
          edgesInCell.push_back(Edge<int>(3, midP));

        for (int first = 0; first < 3; first++)
          for (int second = first + 1; second < 4; second++)
            if ((first != 1 || second != 3) && (first != 0 || second != 2)) {
              if (intermed[first][second][0] != -1)
                edgesInCell.push_back(Edge<int>(intermed[first][second][0], midP));
              if (intermed[first][second][1] != -1)
                edgesInCell.push_back(Edge<int>(intermed[first][second][1], midP));
            }
      }
      // if there is both diagonal intermeds, connect top with all tops,
      // bot with all bots, then top to all bot intermediates
      else if (intermed[0][2][0] != -1 && intermed[0][2][1] != -1) {
        int midB = intermed[0][2][0], midT = intermed[0][2][1];
        //for bot, check 0-3, 2-3, 0-1, 1-2  & Corner 1,3
        //corners
        if (cornerPos[1] >= 0)
          edgesInCell.push_back(Edge<int>(1, midT));
        if (cornerPos[1] < 0)
          edgesInCell.push_back(Edge<int>(1, midB));
        if (cornerPos[3] >= 0)
          edgesInCell.push_back(Edge<int>(3, midT));
        if (cornerPos[3] < 0)
          edgesInCell.push_back(Edge<int>(3, midB));
        //intermeds
        for (int first = 0; first < 3; first++)
          for (int second = first + 1; second < 4; second++)
            if ((first != 1 || second != 3) && (first != 0 || second != 2)) {
              if (intermed[first][second][0] != -1) {
                edgesInCell.push_back(Edge<int>(intermed[first][second][0], midB));
                edgesInCell.push_back(Edge<int>(intermed[first][second][0], midT));
              }
              if (intermed[first][second][1] != -1)
                edgesInCell.push_back(Edge<int>(intermed[first][second][1], midT));
            }
      }
      //in this case 0 and 2 are either both above or below
      //this breaks down into further cases
      else {
        //Note: there should be no case where all points are above or below
        //this algorithm should not execute under that circumstance

        int topBot = (cornerPos[0] > 0 ? 1 : 0);//if the diagonal is across the top, we make a mesh across the top
        //else we make a mesh across the bottom

        if (cornerPos[1] != cornerPos[0] && cornerPos[3] != cornerPos[0])//if both corners are away from the diagonal
        {//create a mesh on the top of the cell
          edgesInCell.push_back(Edge<int>(intermed[0][3][topBot], intermed[0][1][topBot]));
          edgesInCell.push_back(Edge<int>(intermed[0][3][topBot], intermed[1][2][topBot]));
          edgesInCell.push_back(Edge<int>(intermed[0][3][topBot], intermed[2][3][topBot]));
          edgesInCell.push_back(Edge<int>(intermed[0][1][topBot], intermed[1][2][topBot]));
          edgesInCell.push_back(Edge<int>(intermed[2][3][topBot], intermed[1][2][topBot]));
          if (cornerPos[1] == (topBot == 1 ? -1 : 1))//if the diag is on the cell top, we check if the corner is out of the bottom vice versa.
            edgesInCell.push_back(Edge<int>(intermed[0][1][1], intermed[1][2][0]));//make the intermediate face into triangles
          if (cornerPos[3] == (topBot == 1 ? -1 : 1))//if the diag is on the cell top, we check if the corner is out of the bottom vice versa.
            edgesInCell.push_back(Edge<int>(intermed[0][3][1], intermed[2][3][0]));//make the intermediate face into triangles
        } else//at least one has to be
        {
          //triangles from up corner to opposing intermediates
          if (cornerPos[1] == cornerPos[0])//either c1 to c3 intermeds
          {
            edgesInCell.push_back(Edge<int>(1, intermed[0][3][topBot]));
            edgesInCell.push_back(Edge<int>(1, intermed[2][3][topBot]));
          } else//or c3 to c1 intermeds
          {
            edgesInCell.push_back(Edge<int>(3, intermed[0][1][topBot]));
            edgesInCell.push_back(Edge<int>(3, intermed[1][2][topBot]));
          }
        }
      }


      /* point index structure notes: 0 is always top left, 1 is always top right,
       * 2 is bottom right, 3 is bottom left. When intermediates exist, they are created
       * top intermediate first, bottom intermediate second, and are considered in the order
       * 0-1 0-2 0-3 1-2 2-3, So for example. If 0 is below the cell, 1 is above the cell, 2 is
       * in the cell, and 3 is below the cell. Point 4 is a 0-1 intermediate resting on the ceiling
       * of the cell. Point 5 is a 0-1 intermediate resting on the bottom of the cell. Point 6 is a
       * 0-2 intermediate resting on the bottom. Point 7 is a 1-2 intermediate resting on the top, and
       * Point 8 is a 2-3 intermediate resting on the bottom. I'll send you a picture depicting the look
       * of this, along with the edges. Note, the point order matters for indexing, but the order the edges
       * are written in should not matter, edge(0-3) is the same as edge(3-0) and the index of each edge
       * also should not matter.
       */

      terrainPoints.clear();
      terrainEdges.clear();
      for (size_t i = 0; i < pointsInCell.size(); i++)
        terrainPoints.push_back(pointsInCell[i]);
      for (size_t i = 0; i < edgesInCell.size(); i++)
        terrainEdges.push_back(edgesInCell[i]);
      location = Vector3(corners[0][0], corners[0][1], cellBot);
      dimensions = Vector3(corners[1][0] - corners[0][0], corners[0][1] - corners[3][1], dz_array[k]);

      for (int i = 0; i < 6; i++) {
        fluidFacePoints[i].clear();
      }
      // set fluid points for the XZ and YZ faces
      for (int i = 0; i < 4; i++) {
        int firstC, secondC;
        if (i == 0) {
          firstC = 0;
          secondC = 3;
        } else if (i == 1) {
          firstC = 1;
          secondC = 2;
        } else if (i == 2) {
          firstC = 2;
          secondC = 3;
        } else {
          firstC = 0;
          secondC = 1;
        }

        if (pointsInCell[firstC][2] > location[2] + dimensions[2]
            && pointsInCell[secondC][2] > location[2] + dimensions[2]) {
          fluidFacePoints[i].clear();
        } else// if here, then this face is cut
        {
          if (pointsInCell[firstC][2] < location[2] + dimensions[2]) {
            fluidFacePoints[i].push_back(pointsInCell[firstC]);
            fluidFacePoints[i].push_back(Vector3(pointsInCell[firstC][0], pointsInCell[firstC][1], location[2] + dimensions[2]));
          } else if (intermed[firstC][secondC][1] == -1)
            fluidFacePoints[i].push_back(Vector3(pointsInCell[firstC][0], pointsInCell[firstC][1], location[2] + dimensions[2]));

          if (pointsInCell[secondC][2] < location[2] + dimensions[2]) {
            fluidFacePoints[i].push_back(pointsInCell[secondC]);
            fluidFacePoints[i].push_back(Vector3(pointsInCell[secondC][0], pointsInCell[secondC][1], location[2] + dimensions[2]));
          }

          for (int j = 0; j < 2; j++)
            if (intermed[firstC][secondC][j] != -1)
              fluidFacePoints[i].push_back(pointsInCell[intermed[firstC][secondC][j]]);

          if (fluidFacePoints[i].size() <= 2)
            fluidFacePoints[i].clear();
        }
      }

      for (int i = 4; i < 6; i++) {
        for (int j = 0; j < 4; j++)
          if ((i == 4) ? pointsInCell[j][2] <= location[2] :// if this is the bottom, we check if the corner is on the floor
                pointsInCell[j][2] < location[2] + dimensions[2])// else we check if it's under the ceiling
            fluidFacePoints[i].push_back(pointsInCell[j]);

        for (int first = 0; first < 3; first++)
          for (int second = first + 1; second < 4; second++)
            if (first != 1 || second != 3)
              if (intermed[first][second][i - 4] != -1)
                fluidFacePoints[i].push_back(pointsInCell[intermed[first][second][i - 4]]);
        if (fluidFacePoints[i].size() <= 2)
          fluidFacePoints[i].clear();
      }

      count += 1;
      float S_front, S_behind, S_right, S_left, S_below, S_above;
      float S_cut;
      float ni, nj, nk;
      float solid_V_frac;
      float distance_x, distance_y, distance_z;
      float distance;// Distance of cut face from the center of the cut-cell
      std::vector<Vector3> cut_points;// Intersection points for each face

      S_front = S_behind = S_right = S_left = S_below = S_above = 0.0;
      S_cut = 0.0;
      const float pi = 4.0f * atan(1.0); /**< pi constant */
      int kkk = cutcell_index / ((WGD->nx - 1) * (WGD->ny - 1));
      int jjj = (cutcell_index - kkk * (WGD->nx - 1) * (WGD->ny - 1)) / (WGD->nx - 1);
      int iii = cutcell_index - kkk * (WGD->nx - 1) * (WGD->ny - 1) - jjj * (WGD->nx - 1);
      for (int i = 0; i < 6; i++) {

        ni = nj = nk = 0.0;
        solid_V_frac = 0.0;
        cut_points.clear();
        cut_points = fluidFacePoints[i];
        // place points in local cell space
        if (cut_points.size() > 2) {
          for (size_t jj = 0; jj < cut_points.size(); jj++) {
            for (int l = 0; l < 3; l++) {
              cut_points[jj][l] = cut_points[jj][l] - location[l];
            }
          }

          // for faces that exist on the side of the cell (not XY planes)
          if (i < 4) {
            WGD->cut_cell->reorderPoints(cut_points, i, pi);
            if (i == 0) {
              S_right = WGD->cut_cell->calculateArea(cut_points, cutcell_index, WGD->dx, WGD->dy, WGD->dz_array[k], WGD->n, WGD->m, WGD->f, WGD->e, WGD->h, WGD->g, i);
            }
            if (i == 1) {
              S_left = WGD->cut_cell->calculateArea(cut_points, cutcell_index, WGD->dx, WGD->dy, WGD->dz_array[k], WGD->n, WGD->m, WGD->f, WGD->e, WGD->h, WGD->g, i);
            }
            if (i == 2) {
              S_front = WGD->cut_cell->calculateArea(cut_points, cutcell_index, WGD->dx, WGD->dy, WGD->dz_array[k], WGD->n, WGD->m, WGD->f, WGD->e, WGD->h, WGD->g, i);
            }
            if (i == 3) {
              S_behind = WGD->cut_cell->calculateArea(cut_points, cutcell_index, WGD->dx, WGD->dy, WGD->dz_array[k], WGD->n, WGD->m, WGD->f, WGD->e, WGD->h, WGD->g, i);
            }
          }
        } else {
          if (i < 4) {
            if (i == 0) {
              S_right = (WGD->dx * WGD->dz_array[k]);
              WGD->h[cutcell_index] = 0.0;
            }
            if (i == 1) {
              S_left = (WGD->dx * WGD->dz_array[k]);
              WGD->g[cutcell_index] = 0.0;
            }
            if (i == 2) {
              S_front = (WGD->dy * WGD->dz_array[k]);
              WGD->e[cutcell_index] = 0.0;
            }
            if (i == 3) {
              S_behind = (WGD->dy * WGD->dz_array[k]);
              WGD->f[cutcell_index] = 0.0;
            }
          }
        }
        // for the top and bottom faces of the cell (XY planes)

        if (i == 4) {
          S_below = WGD->cut_cell->calculateAreaTopBot(terrainPoints, terrainEdges, cutcell_index, WGD->dx, WGD->dy, WGD->dz_array[k], location, WGD->n, true);
        }

        if (i == 5) {
          S_above = WGD->cut_cell->calculateAreaTopBot(terrainPoints, terrainEdges, cutcell_index, WGD->dx, WGD->dy, WGD->dz_array[k], location, WGD->m, false);
        }

        S_cut = sqrt(pow(S_behind - S_front, 2.0) + pow(S_right - S_left, 2.0) + pow(S_below - S_above, 2.0));

        if (S_cut != 0.0) {
          ni = (S_behind - S_front) / S_cut;
          nj = (S_right - S_left) / S_cut;
          nk = (S_below - S_above) / S_cut;
        }

        if (i == 0) {
          solid_V_frac += (0.0 * (-1) * S_right) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
        }

        if (i == 1) {
          solid_V_frac += (WGD->dy * (1) * S_left) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
        }

        if (i == 2) {
          solid_V_frac += (WGD->dx * (1) * S_front) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
        }

        if (i == 3) {
          solid_V_frac += (0.0 * (-1) * S_behind) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
        }

        if (i == 4) {
          solid_V_frac += (0.0 * (-1) * S_below) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
        }

        if (i == 5) {
          solid_V_frac += (WGD->dz_array[k] * (1) * S_above) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
        }
      }

      if (terrainPoints.size() != 0) {
        solid_V_frac += (((terrainPoints[0][0] - location[0]) * ni * S_cut)
                         + ((terrainPoints[0][1] - location[1]) * nj * S_cut) + ((terrainPoints[0][2] - location[2]) * nk * S_cut))
                        / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
      }

      distance_x = (terrainPoints[0][0] - WGD->x[iii]) * ni;
      distance_y = (terrainPoints[0][1] - WGD->y[jjj]) * nj;
      distance_z = (terrainPoints[0][2] - WGD->z[kkk]) * nk;

      distance = sqrt(pow(distance_x, 2.0) + pow(distance_y, 2.0) + pow(distance_z, 2.0));

      // If the cell center is not in solid
      if (WGD->center_id[cutcell_index] == 1) {
        WGD->wall_distance[cutcell_index] = distance;
      } else {// If the cell center is inside solid
        WGD->wall_distance[cutcell_index] = -distance;
      }

      WGD->terrain_volume_frac[cutcell_index] -= solid_V_frac;

      WGD->ni[cutcell_index] = ni;
      WGD->nj[cutcell_index] = nj;
      WGD->nk[cutcell_index] = nk;

      if (WGD->terrain_volume_frac[cutcell_index] < 0.0) {
        WGD->terrain_volume_frac[cutcell_index] = 0.0;
      }

      if (WGD->terrain_volume_frac[cutcell_index] <= 0.1) {
        WGD->icellflag[cutcell_index] = 2;
        WGD->e[cutcell_index] = 1.0;
        WGD->f[cutcell_index] = 1.0;
        WGD->g[cutcell_index] = 1.0;
        WGD->h[cutcell_index] = 1.0;
        WGD->m[cutcell_index] = 1.0;
        WGD->n[cutcell_index] = 1.0;
      }

      if (WGD->e[cutcell_index] == 0.0 && WGD->f[cutcell_index] == 0.0 && WGD->g[cutcell_index] == 0.0
          && WGD->h[cutcell_index] == 0.0 && WGD->m[cutcell_index] == 0.0 && WGD->n[cutcell_index] == 0.0) {
        WGD->icellflag[cutcell_index] = 2;
        WGD->e[cutcell_index] = 1.0;
        WGD->f[cutcell_index] = 1.0;
        WGD->g[cutcell_index] = 1.0;
        WGD->h[cutcell_index] = 1.0;
        WGD->m[cutcell_index] = 1.0;
        WGD->n[cutcell_index] = 1.0;
      }
    }
  }
}

Vector3 DTEHeightField::getIntermediate(Vector3 a, Vector3 b, float height) const
{

  if (a[2] == b[2])
    return Vector3((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, a[2]);

  float offset = a[2] - height;
  float xCoord, yCoord;
  if (a[0] != b[0]) {
    float slopeX;
    slopeX = (b[2] - a[2]) / (b[0] - a[0]);
    float xOff = fabs((offset) / slopeX);
    xCoord = a[0] < b[0] ? a[0] + xOff : a[0] - xOff;

  } else
    xCoord = a[0];

  if (a[1] != b[1]) {
    float slopeY;
    slopeY = (b[2] - a[2]) / (b[1] - a[1]);
    float yOff = fabs((offset) / slopeY);
    yCoord = a[1] < b[1] ? a[1] + yOff : a[1] - yOff;
  } else
    yCoord = a[1];

  return Vector3(xCoord, yCoord, height);
}

void DTEHeightField::closeScanner()
{
  CPLFree(pafScanline);
}
