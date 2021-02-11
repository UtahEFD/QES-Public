#include "DTEHeightField.h"

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
                               std::tuple<float, float, float> cellSize, float UTMx, float UTMy)
  : m_filename(filename), m_rbMin(0.0),
    m_cellSize( cellSize ), m_dim( dim ),
    domain_UTMx(UTMx), domain_UTMy(UTMy)
{
    GDALAllRegister();

  // only for texture???
 // loadImage();

  load();
}

// Constructor for converting heightfield data to the internal
// representation for digital elevation
//
// Inputs need to provide the nx, ny, nz
DTEHeightField::DTEHeightField(const std::vector<double> &heightField,
                               std::tuple<int, int, int> dim,
                               std::tuple<float, float, float> cellSize,
                               float halo_x, float halo_y)
    : m_cellSize( cellSize ), m_dim( dim )
{
    Triangle *tPtr=0;
    m_triList.clear();

    // local variables to hold triangle vertices
    // Vector3<float> tc0, tc1, tc2;

    // local variables to hold common variables
    auto [ nx, ny, nz ] = m_dim;
    auto [ dx, dy, dz ] = m_cellSize;
    
    std::cout << "Loading digital elevation data from height field\n";

    std::cout << "dimX = " << nx << ", dimY = " << ny << ", dimZ = " << nz << std::endl;
    std::cout << "cellSizes = (" << dx << ", " << dy << ", " << dz << ")" << std::endl;
    
    std::cout << "size of heightField = " << heightField.size() << std::endl;

    // eventually need the fm_dx and fm_dy so we can multiply into
    // correct dimensions

    int step = 1;  // step size is interpretted incorrectly here for
                   // fire meshes...

    // previously, with regular DEMs we can use the cellSize to
    // determine how we step over the terrain to create the actual
    // mesh... based on dx, dy...

    // This triangle mesh is in the dimensions of the height field
    // array... may not be in the domain space... hence when queried
    // later in the

    std::cout << "Adding halo to surrounding regions..." << halo_x << ", " << halo_y << std::endl;

    for (float j = 0; j < ny-1; j+=step) {
        for (float i = 0; i < nx-1; i+=step) {

            int idx = j * nx + i;
            if (idx > heightField.size() - 1) idx = heightField.size()-1;
            // std::cout << "(" << i << ", " << j << ") = " << heightField[idx] << std::endl;

            // when pulling data from the height field and converting
            // to actual locations, we need to add the halo_x and
            // halo_y to all positions to shift the domain -- these
            // are in meters...

            //
            // Need to convert these to localized QES dimensions
            //
            float xPos = halo_x + (i * dx);
            float yPos = halo_y + (j * dy);

            // Vector3<float> tv0( i, j, (float)heightField[ idx ] ); // queryHeight( pafScanline, Xpixel,  Yline));
            Vector3<float> tv0( xPos, yPos, (float)heightField[ idx ] ); // queryHeight( pafScanline, Xpixel,  Yline));

            idx = j * nx + (i + step);
            if (idx > heightField.size() - 1) { std::cout << "***************" << std::endl; idx = heightField.size()-1; }
            
            // Vector3<float> tv1( i+step, j, (float)heightField[ idx ] ); // queryHeight( pafScanline,  (int)(iXpixel + stepX ), Yline ) );
            xPos = halo_x + ((i+step) * dx);
            Vector3<float> tv1( xPos, yPos, (float)heightField[ idx ] ); // queryHeight( pafScanline,  (int)(iXpixel + stepX ), Yline ) );

            idx = (j+step) * nx + i;
            if (idx > heightField.size() - 1) { std::cout << "---------------" << std::endl; idx = heightField.size()-1; }
            
            // Vector3<float> tv2( i, j+step, (float)heightField[ idx] ); // queryHeight( pafScanline, Xpixel, (int)(iYline + stepY) ));
            xPos = halo_x + (i * dx);
            yPos = halo_y + ((j+step) * dy);
            Vector3<float> tv2( xPos, yPos, (float)heightField[ idx] ); // queryHeight( pafScanline, Xpixel, (int)(iYline + stepY) ));

            // std::cout << "Triangle: (" << tv0[0] << ", " << tv0[1] << ", " << tv0[2] <<  "), (" << tv1[0] << ", " << tv1[1] << ", " << tv1[2] <<  "), (" << tv2[0] << ", " << tv2[1] << ", " << tv2[2] <<  ")" << std::endl;

            tPtr = new Triangle( tv0, tv1, tv2 );
            m_triList.push_back(tPtr);

            idx = (j+step) * nx + i;
            if (idx > heightField.size() - 1) idx = heightField.size()-1;
            
            // Vector3<float> tv3( i, j+step, (float)heightField[ idx ] );// queryHeight( pafScanline,  Xpixel, (int)(iYline + stepY) ) );
            xPos = halo_x + (i * dx);
            yPos = halo_y + ((j+step) * dy);
            Vector3<float> tv3( xPos, yPos, (float)heightField[ idx ] );// queryHeight( pafScanline,  Xpixel, (int)(iYline + stepY) ) );            

            idx = j * nx + (i+step);
            if (idx > heightField.size() - 1) idx = heightField.size()-1;
            // Vector3<float> tv4( i+step, j, (float)heightField[ idx ] ); //  queryHeight( pafScanline,  (int)(iXpixel + stepX) , Yline ) );
            xPos = halo_x + ((i+step) * dx);
            yPos = halo_y + (j * dy);
            Vector3<float> tv4( xPos, yPos, (float)heightField[ idx ] ); //  queryHeight( pafScanline,  (int)(iXpixel + stepX) , Yline ) );

            idx = (j+step) * nx + (i+step);
            if (idx > heightField.size() - 1) idx = heightField.size()-1;
            // Vector3<float> tv5( i+step, j+step, (float)heightField[ idx ] ); // queryHeight( pafScanline, (int)(iXpixel + stepX), (int)(iYline + stepY) ) );
            xPos = halo_x + ((i+step) * dx);
            yPos = halo_y + ((j+step) * dy);
            Vector3<float> tv5( xPos, yPos, (float)heightField[ idx ] ); // queryHeight( pafScanline, (int)(iXpixel + stepX), (int)(iYline + stepY) ) );

            // std::cout << "Triangle: (" << tv3[0] << ", " << tv3[1] << ", " << tv3[2] <<  "), (" << tv4[0] << ", " << tv4[1] << ", " << tv4[2] <<  "), (" << tv5[0] << ", " << tv5[1] << ", " << tv5[2] <<  ")" << std::endl;

            tPtr = new Triangle( tv3, tv4, tv5 );
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
  // High res giant image
  // std::string filename = "/scratch/PSP_003910_1685_RED_A_01_ORTHO.JP2";

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
  std::cout << "load: loading DTE..." << std::endl;

  // local variables to hold common variables
  auto [ nx, ny, nz ] = m_dim;
  auto [ dx, dy, dz ] = m_cellSize;

  // From -- http://www.gdal.org/gdal_tutorial.html
  m_poDataset = (GDALDataset *) GDALOpen( m_filename.c_str(), GA_ReadOnly );
  if( m_poDataset == NULL )
    {
      std::cerr << "Couldn't open file: " << m_filename << std::endl;
      exit( EXIT_FAILURE );
    }

  printf( "GDAL Driver: %s/%s\n",
	  m_poDataset->GetDriver()->GetDescription(),
	  m_poDataset->GetDriver()->GetMetadataItem( GDAL_DMD_LONGNAME ) );

  printf( "\tRaster Size is %dx%dx%d\n",
	  m_poDataset->GetRasterXSize(), m_poDataset->GetRasterYSize(),
	  m_poDataset->GetRasterCount() );

  // Attempt to get the spatial reference from this dataset -
  // which will help us convert into lat/long
  // In GDAL 3.0+ we can use
  // spatialRef = m_poDataset->GetSpatialRef, but in pre-3.0 versions,
  // this comes from GetProjectionRef
  if( m_poDataset->GetProjectionRef() != NULL )
    printf( "\tProjection is `%s'\n", m_poDataset->GetProjectionRef() );

  if( m_poDataset->GetGeoTransform( m_geoTransform ) == CE_None )
    {
      printf( "\tOrigin = (%.6f,%.6f)\n",
	      m_geoTransform[0], m_geoTransform[3] );

      printf( "\tPixel Size = (%.6f,%.6f)\n",
	      m_geoTransform[1], m_geoTransform[5] );
      pixelSizeX = abs(m_geoTransform[1]);
      pixelSizeY = abs(m_geoTransform[5]);

      printf( "These should be zero for north up: (%.6f, %.6f)\n",
	      m_geoTransform[2], m_geoTransform[4] );
    }

  GDALRasterBand  *poBand;
  int             nBlockXSize, nBlockYSize;
  int             bGotMin, bGotMax;
  //double          adfMinMax[2];

  poBand = m_poDataset->GetRasterBand( 1 );
  poBand->GetBlockSize( &nBlockXSize, &nBlockYSize );
  printf( "\tRaster Block=%dx%d Type=%s, ColorInterp=%s\n",
	  nBlockXSize, nBlockYSize,
	  GDALGetDataTypeName(poBand->GetRasterDataType()),
	  GDALGetColorInterpretationName(poBand->GetColorInterpretation()) );

  adfMinMax[0] = poBand->GetMinimum( &bGotMin );
  adfMinMax[1] = poBand->GetMaximum( &bGotMax );
  if( ! (bGotMin && bGotMax) )
    GDALComputeRasterMinMax((GDALRasterBandH)poBand, TRUE, adfMinMax);

  printf( "\tRaster Min=%.3fd and Max=%.3f\n", adfMinMax[0], adfMinMax[1] );

  if( poBand->GetOverviewCount() > 0 )
    printf( "\tBand has %d overviews.\n", poBand->GetOverviewCount() );

  if( poBand->GetColorTable() != NULL )
    printf( "\tBand has a color table with %d entries.\n",
	    poBand->GetColorTable()->GetColorEntryCount() );

  m_rbScale = poBand->GetScale();
  printf( "Band has scale: %.4f\n", m_rbScale );

  m_rbOffset = poBand->GetOffset();
  printf( "Band has offset: %.4f\n", m_rbOffset );

  m_rbNoData = poBand->GetNoDataValue();
  printf( "Band has NoData value: %.4f\n", m_rbNoData );

  m_nXSize = poBand->GetXSize();
  m_nYSize = poBand->GetYSize();

  origin_x = m_geoTransform[0];
  origin_y = m_geoTransform[3] - pixelSizeY*m_nYSize;

  printf( "DEM size is %dx%dx%d\n",
	  m_nXSize, m_nYSize );

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
  convertRasterToGeo( 0, 0, xGeo, yGeo );
  printf("Raster Coordinate (0, 0):\t(%12.7f, %12.7f)\n", xGeo, yGeo);

  convertRasterToGeo( m_nXSize, 0, xGeo, yGeo );
  printf("Raster Coordinate (%d, 0):\t(%12.7f, %12.7f)\n", m_nXSize, xGeo, yGeo);

  convertRasterToGeo( m_nXSize, m_nYSize, xGeo, yGeo );
  printf("Raster Coordinate (%d, %d):\t(%12.7f, %12.7f)\n", m_nXSize, m_nYSize, xGeo, yGeo);

  convertRasterToGeo( 0, m_nYSize, xGeo, yGeo );
  printf("Raster Coordinate (0, %d):\t(%12.7f, %12.7f)\n", m_nYSize, xGeo, yGeo);

  if ( (domain_UTMx > origin_x) || (domain_UTMy > origin_y) )
  {
  	shift_x = (domain_UTMx-origin_x)/pixelSizeX;
  	shift_y = (domain_UTMy-origin_y)/pixelSizeY;
  }

  float domain_end_x = domain_UTMx + nx*dx;
  float domain_end_y = domain_UTMy + ny*dy;
  float dem_end_x = origin_x + m_nXSize*pixelSizeX;
  float dem_end_y = origin_y + m_nYSize*pixelSizeY;

  if ( (domain_end_x < dem_end_x) || (domain_end_y < dem_end_y) )
  {
  	end_x = (dem_end_x-domain_end_x)/pixelSizeX;
  	end_y = (dem_end_y-domain_end_y)/pixelSizeY;
  }

  pafScanline = (float *) CPLMalloc(sizeof(float)*(m_nXSize)*(m_nYSize));

  m_nXSize = m_nXSize - shift_x - end_x;
  m_nYSize = m_nYSize - shift_y - end_y;


  // rb->RasterIO(GF_Read, 0, 0, xsize, ysize, buffer, xsize, ysize, GDT_Float32, 0, 0);
  //
  // CPLErr - CE_Failure if the access fails, otherwise CE_None.
  CPLErr rasterErr = poBand->RasterIO( GF_Read, shift_x, end_y,
                                       m_nXSize, m_nYSize,
                                       pafScanline,
                                       m_nXSize, m_nYSize, GDT_Float32,
                                       0, 0 );
  if (rasterErr == CE_Failure) {
      std::cerr << "CPL RasterIO failure during DEM loading. Exiting." << std::endl;
      exit(EXIT_FAILURE);
  }


  Triangle *tPtr=0;
  m_triList.clear();

  //double xGeo, yGeo;
  Vector3<float> tc0, tc1, tc2;


  std::cout << "DEM Loading\n";

  float stepX = dx / pixelSizeX; // tie back to dx, dy here.... with scaling of pixelsize
  float stepY = dy / pixelSizeY;

  assert(stepX > 0 && stepY > 0);

  for (float iYline = 0; iYline < m_nYSize-1; iYline+=stepY) {
    for (float iXpixel = 0; iXpixel < m_nXSize-1; iXpixel+=stepX) {

      int Yline = (int)iYline;
      int Xpixel = (int)iXpixel;


      // For these purposes, pixel refers to the "X" coordinate, while
      // line refers to the "Z" coordinate

      // turn localized coordinates (Yline and Xpixel) into geo-referenced values.
      // then use the geo referenced coordinate to lookup the height.


//      xGeo = m_geoTransform[0] + Xpixel * m_geoTransform[1];
//      yGeo = m_geoTransform[3] + Yline * m_geoTransform[5];

//height set to y coord, change to z

    //These "should" be real unit-based triangles.. hopefully meters..
      Vector3<float> tv0( iXpixel * pixelSizeX,    iYline * pixelSizeY,     queryHeight( pafScanline, Xpixel,  Yline));
      Vector3<float> tv1( (iXpixel + stepX) * pixelSizeX, iYline * pixelSizeY, queryHeight( pafScanline,  (int)(iXpixel + stepX ), Yline ) );
      Vector3<float> tv2( iXpixel * pixelSizeX, (iYline + stepY) * pixelSizeY, queryHeight( pafScanline, Xpixel, (int)(iYline + stepY) ));

      tPtr = new Triangle( tv0, tv1, tv2 );
      m_triList.push_back(tPtr);


      Vector3<float> tv3(  iXpixel * pixelSizeX, (iYline + stepY) * pixelSizeY, queryHeight( pafScanline,  Xpixel, (int)(iYline + stepY) ) );
     // convertToTexCoord(Yline+step, Xpixel, tc0);

      Vector3<float> tv4(  (iXpixel + stepX) * pixelSizeX, iYline * pixelSizeY,  queryHeight( pafScanline,  (int)(iXpixel + stepX) , Yline ) );
      // convertToTexCoord(Yline, Xpixel+step, tc1);

      Vector3<float> tv5(  (iXpixel + stepX) * pixelSizeX,(iYline + stepY) * pixelSizeY, queryHeight( pafScanline, (int)(iXpixel + stepX), (int)(iYline + stepY) ) );
      // convertToTexCoord(Yline+step, Xpixel+step, tc2);

      tPtr = new Triangle( tv3, tv4, tv5 );
      m_triList.push_back(tPtr);
    }
    printProgress((iYline / (float)m_nYSize));
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

void DTEHeightField::setDomain(Vector3<int>* domain, Vector3<float>* grid)
{
    for (int i = 0; i < 3; i++)
    {
      min[i] = LIMIT;
      max[i] = 0.0f;
    }

    auto start = std::chrono::high_resolution_clock::now(); // Start
                                                                 // recording
                                                                 // execution
                                                                 // time

    std::cout << "Setting Terrain Boundaries\n";
    for (int q = 0; q < 3; q++)
    {
        // if (q == 0)
        // std::cout << "in X...";
        // else if (q == 1)
        // std::cout << "in Y...";
        // else
        // std::cout << "in Z...";

      int triListSize = m_triList.size();

#pragma acc parallel loop
      for (int i = 0; i < triListSize; i++)
      {
        if ( (*(m_triList[i]->a))[q] >= 0 && (*(m_triList[i]->a))[q] < min[q] )
          min[q] = (*(m_triList[i]->a))[q];
        if ( (*(m_triList[i]->b))[q] >= 0 && (*(m_triList[i]->b))[q] < min[q] )
          min[q] = (*(m_triList[i]->b))[q];
        if ( (*(m_triList[i]->c))[q] >= 0 && (*(m_triList[i]->c))[q] < min[q] )
          min[q] = (*(m_triList[i]->c))[q];

        if ( (*(m_triList[i]->a))[q] > max[q] && (*(m_triList[i]->a))[q] < LIMIT)
          max[q] = (*(m_triList[i]->a))[q];
        if ( (*(m_triList[i]->b))[q] > max[q] && (*(m_triList[i]->b))[q] < LIMIT)
          max[q] = (*(m_triList[i]->b))[q];
        if ( (*(m_triList[i]->c))[q] > max[q] && (*(m_triList[i]->c))[q] < LIMIT)
          max[q] = (*(m_triList[i]->c))[q];
      }

#pragma acc parallel loop
      for (int i = 0; i < triListSize; i++)
      {
        (*(m_triList[i]->a))[q] -= min[q];
        (*(m_triList[i]->b))[q] -= min[q];
        (*(m_triList[i]->c))[q] -= min[q];
      }

      /*if (q != 2)
      {
        max[q] -= min[q];
        (*domain)[q] = (int)(max[q] / (float)(*grid)[q]) + 1;
      }
      else
        max[q] = max[q] - min[q] + (float)((*grid)[2]);

      printf ("max %lf grid %lf\n" , max[q], (float)(*grid)[q]);*/

      //current implementation adds buffer in z dim for buffer space
      //get more specific values, currently adding 50 meters
      //Also, domains are currently only working with cubic dimensions... fix this

      /*if (q == 2)
        (*domain)[q] += (int)(50.0f / (float)(*grid)[q]);*/

      // std::cout << " done." << std::endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();  // Finish recording execution time

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "\telapsed time: " << elapsed.count() << " s\n";   // Print out elapsed execution time

    /* if ((*domain)[0] >= (*domain)[1] && (*domain)[0] >= (*domain)[2])
      (*domain)[1] = (*domain) [2] = (*domain)[0];
    else if ((*domain)[1] >= (*domain)[0] && (*domain)[1] >= (*domain)[2])
      (*domain)[0] = (*domain) [2] = (*domain)[1];
    else
        (*domain)[0] = (*domain) [1] = (*domain)[2]; */
    printf("Newly calculated domain size: %d %d %d\n", (*domain)[0], (*domain)[1], (*domain)[2]);
}

void DTEHeightField::outputOBJ(std::string s)
{
  std::ofstream file;
  file.open(s.c_str());

  std::vector< Vector3<float>* > verts;
  std::vector< Vector3<int>* > tris;

  for (int i = 0; i < m_triList.size(); i++)
  {

    Triangle* t = m_triList[i];

    Vector3<int> *tVs = new Vector3<int>(-1, -1 , -1);

    for (int j = 0; j < verts.size(); j++)
    {
      if ( (*(t->a)) == (*verts[j]) )
        (*tVs)[0] = j + 1;
      if ( (*(t->b)) == (*verts[j]) )
        (*tVs)[1] = j + 1;
      if ( (*(t->c)) == (*verts[j]) )
        (*tVs)[2] = j + 1;
    }

    if ((*tVs)[0] == -1)
    {
      verts.push_back( t->a );
      (*tVs)[0] = verts.size();
    }

    if ((*tVs)[1] == -1)
    {
      verts.push_back( t->b );
      (*tVs)[1] = verts.size();
    }

    if ((*tVs)[2] == -1)
    {
      verts.push_back( t->c );
      (*tVs)[2] = verts.size();
    }

    tris.push_back(tVs);
    printProgress( ( (float)i / (float)m_triList.size()) * (9.0f / 10.0f) );
  }

  for(int i = 0; i < verts.size(); i++)
  {
    file << "v " << (*(verts[i]))[0] << " " << (*(verts[i]))[1] << " " << (*(verts[i]))[2] << "\n";
    printProgress( ( (float)i / (float)m_triList.size()) / 20.0f + 0.9f );
  }

  for(int i = 0; i < tris.size(); i++)
  {
    file << "f " << (*(tris[i]))[0] << " " << (*(tris[i]))[1] << " " << (*(tris[i]))[2] << "\n";
    printProgress( ( (float)i / (float)m_triList.size()) / 20.0f + 0.95f);
  }

  file.close();
}

void DTEHeightField::printProgress (float percentage)
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}



#define CELL(i,j,k) ((i) + (j) * (nx - 1) + (k) * (nx - 1) * (ny - 1))
#define CLAMP(low, high, x) ( (x) < (low) ? (low) : ( (x) > (high) ? (high) : (x) ))

std::vector<int> DTEHeightField::setCells(Cell* cells, int nx, int ny, int nz, float dx, float dy, std::vector<float> &dz_array, std::vector<float> z_face, float halo_x, float halo_y) const
{

  printf("Setting Cell Data...\n");
  auto start = std::chrono::high_resolution_clock::now(); // Start recording execution time

  std::vector<int> cutCells;

  int ii = halo_x/dx;
	int jj = halo_y/dy;
	int i_domain_end = ii+(m_nXSize*pixelSizeX)/dx;
	int j_domain_end = jj+(m_nYSize*pixelSizeY)/dy;

  for (int i = 0; i < nx - 2; i++)
    for (int j = 0; j < ny - 2; j++)
    {

      //all work here is done for each column of cells in the z direction from the xy plane.

       Vector3<float> corners[4]; //stored from top Left in clockwise order
       if (i >= ii && j >= jj && i <= i_domain_end && j <= j_domain_end)
       {
         corners[0] = Vector3<float>( i * dx, j * dy, CLAMP(0, max[2], queryHeight( pafScanline , ((i-ii) * dx) / pixelSizeX,  ( (j-jj) * dy) / pixelSizeY) - min[2]) );
         corners[1] = Vector3<float>( i * dx, (j+1) * dy, CLAMP(0, max[2], queryHeight( pafScanline,  ( (i-ii) * dx)/ pixelSizeX, (((j-jj) + 1) * dy) / pixelSizeY) - min[2]) );
         corners[2] = Vector3<float>( (i + 1) * dx, (j + 1) * dy, CLAMP(0, max[2], queryHeight( pafScanline , (((i-ii) + 1) * dx) / pixelSizeX,  (((j-jj) + 1) * dy) / pixelSizeY) - min[2]) );
         corners[3] = Vector3<float>( (i + 1) * dx, j * dy, CLAMP(0, max[2], queryHeight( pafScanline, (((i-ii) + 1) * dx) / pixelSizeX, ((j-jj) * dy) / pixelSizeY ) - min[2]) );
       }
       else
       {
         corners[0] = Vector3<float>( i * dx, j * dy,   0.0f);
         corners[1] = Vector3<float>( i * dx, (j + 1) * dy, 0.0f);
         corners[2] = Vector3<float>( (i + 1) * dx, (j + 1) * dy, 0.0f);
         corners[3] = Vector3<float>( (i + 1) * dx, j * dy, 0.0f);
       }
       /*else if (i < ii)
       {
         if (j < jj)
         {
           //std::cout << "height:  " << queryHeight( pafScanline , ( dx) / pixelSizeX,  ( dy) / pixelSizeY) - min[2] << std::endl;
           corners[0] = Vector3<float>( i * dx, j * dy,   queryHeight( pafScanline , ( dx) / pixelSizeX,  ( dy) / pixelSizeY) - min[2]);
           corners[1] = Vector3<float>( i * dx, (j + 1) * dy, queryHeight( pafScanline , ( dx) / pixelSizeX,  ( dy) / pixelSizeY) - min[2]);
           corners[2] = Vector3<float>( (i + 1) * dx, (j + 1) * dy, queryHeight( pafScanline , (dx) / pixelSizeX,  (dy) / pixelSizeY) - min[2]);
           corners[3] = Vector3<float>( (i + 1) * dx, j * dy, queryHeight( pafScanline , (dx) / pixelSizeX,  (dy) / pixelSizeY) - min[2]);
         }
         else if (j > j_domain_end)
         {
           corners[0] = Vector3<float>( i * dx, j * dy,   queryHeight( pafScanline , ( dx) / pixelSizeX,  ( (j_domain_end-jj-1) * dy) / pixelSizeY) - min[2]);
           corners[1] = Vector3<float>( i * dx, (j + 1) * dy, queryHeight( pafScanline , (dx) / pixelSizeX,  ( (j_domain_end-jj-1) * dy) / pixelSizeY) - min[2]);
           corners[2] = Vector3<float>( (i + 1) * dx, (j + 1) * dy, queryHeight( pafScanline , (dx) / pixelSizeX,  ( (j_domain_end-jj-1) * dy) / pixelSizeY) - min[2]);
           corners[3] = Vector3<float>( (i + 1) * dx, j * dy, queryHeight( pafScanline , (dx) / pixelSizeX,  ( (j_domain_end-jj-1) * dy) / pixelSizeY) - min[2]);
         }
         else
         {
           corners[0] = Vector3<float>( i * dx, j * dy,   queryHeight( pafScanline , (dx) / pixelSizeX,  ( (j-jj) * dy) / pixelSizeY) - min[2]);
           corners[1] = Vector3<float>( i * dx, (j + 1) * dy, queryHeight( pafScanline , (dx) / pixelSizeX,  ( (j-jj) * dy) / pixelSizeY) - min[2]);
           corners[2] = Vector3<float>( (i + 1) * dx, (j + 1) * dy, queryHeight( pafScanline , (dx) / pixelSizeX,  ( (j-jj) * dy) / pixelSizeY) - min[2]);
           corners[3] = Vector3<float>( (i + 1) * dx, j * dy, queryHeight( pafScanline , (dx) / pixelSizeX,  ( (j-jj) * dy) / pixelSizeY) - min[2]);
         }
       }

       else if (j < jj)
       {
         if (i > i_domain_end)
         {
           corners[0] = Vector3<float>( i * dx, j * dy,   queryHeight( pafScanline , ( (i_domain_end-ii-1) * dx) / pixelSizeX,  (dy) / pixelSizeY) - min[2]);
           corners[1] = Vector3<float>( i * dx, (j + 1) * dy, queryHeight( pafScanline , ( (i_domain_end-ii-1) * dx) / pixelSizeX,  (dy) / pixelSizeY) - min[2]);
           corners[2] = Vector3<float>( (i + 1) * dx, (j + 1) * dy, queryHeight( pafScanline , ( (i_domain_end-ii-1) * dx) / pixelSizeX,  (dy) / pixelSizeY) - min[2]);
           corners[3] = Vector3<float>( (i + 1) * dx, j * dy, queryHeight( pafScanline , ( (i_domain_end-ii-1) * dx) / pixelSizeX,  (dy) / pixelSizeY) - min[2]);
         }
         else
         {
           corners[0] = Vector3<float>( i * dx, j * dy,   queryHeight( pafScanline , ((i-ii) * dx) / pixelSizeX,  (dy) / pixelSizeY) - min[2]);
           corners[1] = Vector3<float>( i * dx, (j + 1) * dy, queryHeight( pafScanline , ((i-ii) * dx) / pixelSizeX,  (dy) / pixelSizeY) - min[2]);
           corners[2] = Vector3<float>( (i + 1) * dx, (j + 1) * dy, queryHeight( pafScanline , ((i-ii) * dx) / pixelSizeX,  (dy) / pixelSizeY) - min[2]);
           corners[3] = Vector3<float>( (i + 1) * dx, j * dy, queryHeight( pafScanline , ((i-ii) * dx) / pixelSizeX,  (dy) / pixelSizeY) - min[2]);
         }
       }

       else if (i > i_domain_end)
       {
         if (j > j_domain_end)
         {
           corners[0] = Vector3<float>( i * dx, j * dy,   queryHeight( pafScanline , ( (i_domain_end-ii-1) * dx) / pixelSizeX,  ( (j_domain_end-jj-1) * dy) / pixelSizeY) - min[2]);
           corners[1] = Vector3<float>( i * dx, (j + 1) * dy, queryHeight( pafScanline , ( (i_domain_end-ii-1) * dx) / pixelSizeX,  ( (j_domain_end-jj-1) * dy) / pixelSizeY) - min[2]);
           corners[2] = Vector3<float>( (i + 1) * dx, (j + 1) * dy, queryHeight( pafScanline , ((i_domain_end-ii-1) * dx) / pixelSizeX,  ( (j_domain_end-jj-1) * dy) / pixelSizeY) - min[2]);
           corners[3] = Vector3<float>( (i + 1) * dx, j * dy, queryHeight( pafScanline , ((i_domain_end-ii-1) * dx) / pixelSizeX,  ( (j_domain_end-jj-1) * dy) / pixelSizeY) - min[2]);
         }
         else
         {
           corners[0] = Vector3<float>( i * dx, j * dy,   queryHeight( pafScanline , ((i_domain_end-ii-1) * dx) / pixelSizeX,  ( (j-jj) * dy) / pixelSizeY) - min[2]);
           corners[1] = Vector3<float>( i * dx, (j + 1) * dy, queryHeight( pafScanline , ( (i_domain_end-ii-1) * dx) / pixelSizeX,  ( (j-jj) * dy) / pixelSizeY) - min[2]);
           corners[2] = Vector3<float>( (i + 1) * dx, (j + 1) * dy, queryHeight( pafScanline , ((i_domain_end-ii-1) * dx) / pixelSizeX,  ( (j-jj) * dy) / pixelSizeY) - min[2]);
           corners[3] = Vector3<float>( (i + 1) * dx, j * dy, queryHeight( pafScanline , ((i_domain_end-ii-1) * dx) / pixelSizeX,  ( (j-jj) * dy) / pixelSizeY) - min[2]);
         }
       }

       else if (j > j_domain_end)
       {
         corners[0] = Vector3<float>( i * dx, j * dy,   queryHeight( pafScanline , ((i-ii) * dx) / pixelSizeX,  ( (j_domain_end-jj-1) * dy) / pixelSizeY) - min[2]);
         corners[1] = Vector3<float>( i * dx, (j + 1) * dy, queryHeight( pafScanline , ((i-ii) * dx) / pixelSizeX,  ( (j_domain_end-jj-1) * dy) / pixelSizeY) - min[2]);
         corners[2] = Vector3<float>( (i + 1) * dx, (j + 1) * dy, queryHeight( pafScanline , ((i-ii) * dx) / pixelSizeX,  ( (j_domain_end-jj-1) * dy) / pixelSizeY) - min[2]);
         corners[3] = Vector3<float>( (i + 1) * dx, j * dy, queryHeight( pafScanline , ((i-ii) * dx) / pixelSizeX,  ( (j_domain_end-jj-1) * dy) / pixelSizeY) - min[2]);
       }

       if (i==319 && j==10)
       {
         std::cout << "((i-ii) + 1) * dx) / pixelSizeX:  " << ((i-ii) + 1) * dx / pixelSizeX << std::endl;
         std::cout << "((j-jj) * dy) / pixelSizeY:  " << ((j-jj) * dy) / pixelSizeY << std::endl;
         std::cout << "(j_domain_end-jj-1):  " << (j_domain_end-jj-1) << std::endl;
         std::cout << "corners[0]:  " << corners[0][2] << std::endl;
         std::cout << "corners[1]:  " << corners[1][2] << std::endl;
         std::cout << "corners[2]:  " << corners[2][2] << std::endl;
         std::cout << "corners[3]:  " << corners[3][2] << std::endl;
       }*/

       setCellPoints(cells, i, j, nx, ny, nz, dz_array, z_face, corners, cutCells);

    }


  auto finish = std::chrono::high_resolution_clock::now();  // Finish recording execution time
  std::chrono::duration<float> elapsed = finish - start;
  std::cout << "Elapsed time For CellSet: " << elapsed.count() << " s\n";   // Print out elapsed execution time

  return cutCells;


}

void DTEHeightField::setCellPoints(Cell* cells, int i, int j, int nx, int ny, int nz, std::vector<float> &dz_array, std::vector<float> z_face, Vector3<float> corners[], std::vector<int>& cutCells) const
{
   float coordsMin, coordsMax;
   coordsMin = coordsMax = corners[0][2];
	for (int l = 1; l <= 3; l++)
	{
		if (coordsMin > corners[l][2])
		{
			coordsMin = corners[l][2];
		}
		else if (coordsMax < corners[l][2])
		{
			coordsMax = corners[l][2];
		}
	}

// #pragma acc parallel loop
  for (int k = 1; k < nz - 1; k++)
  {
    float cellBot = z_face[k-1];
    float cellTop = cellBot + dz_array[k];

    /*if ( i < 5  || j < 5)
    {
      std::cout << "i: " << i << "\t\t" << "cellBot:  " << cellBot << std::endl;
      std::cout << "j: " << j << "\t\t" << "cellTop:  " << cellTop << std::endl;
      std::cout << "k: " << k << "\t\t" << "coordsMin:  " << coordsMin << std::endl;
    }*/

    if ( cellTop <= coordsMin)
      cells[CELL(i,j,k)] = Cell(terrain_CT, Vector3<float>(corners[0][0], corners[0][1], cellBot),
                                            Vector3<float>(corners[1][0] - corners[0][0], corners[0][1] - corners[3][1], dz_array[k]));
    else if ( cellBot >= coordsMax)
      cells[CELL(i,j,k)] = Cell(air_CT, Vector3<float>(corners[0][0], corners[0][1], cellBot),
                                            Vector3<float>(corners[1][0] - corners[0][0], corners[0][1] - corners[3][1], dz_array[k]));
    else
    {
      cutCells.push_back(CELL(i,j,k));
	 /* std::cout << "i:" << i <<"\n";
	  std::cout << "j:" << j <<"\n";
	  std::cout << "k:" << k <<"\n";
	  std::cout << "Number of cutcells:" << cutCells.size()<<"\n";*/
      std::vector< Vector3<float> > pointsInCell;
      std::vector< Edge< int > > edgesInCell;

      //Check to see the positions of the corners, the corners are always in
      //the cell, no matter what. If they exist out of bounds of the cell in the
      //Z dimension, we add them at the floor or ceiling of the cell. We can use
      //this to identify where the actual geometry of the terrain crosses the cell,
      //as anything below the mesh of points and edges is terrain, and above is air.

      int cornerPos[4] = {0, 0, 0, 0}; // 0 is in, 1 is above, -1 is below
      //check if corners are in
      for (int l = 0; l < 4; l++)
        if (corners[l][2] >= cellBot && corners[l][2] <= cellTop)
        {
          pointsInCell.push_back(corners[l]);
          cornerPos[l] = 0;
        }
        else if (corners[l][2] < cellBot)
        {
          cornerPos[l] = -1;
          pointsInCell.push_back( Vector3<float>(corners[l][0], corners[l][1], cellBot) );
        }
        else
        {
          cornerPos[l] = 1;
          pointsInCell.push_back( Vector3<float>(corners[l][0], corners[l][1], cellTop) );
        }
         for (int first = 0; first < 3; first++)
              for (int second = first + 1; second < 4; second++)
                if ( (first != 1 || second != 3) && (first != 0 || second != 2) )
                  if (cornerPos[first] == cornerPos[second])
                    edgesInCell.push_back(Edge<int>(first, second));

      //check intermediates 0-1 0-2 0-3 1-2 2-3
        int intermed[4][4][2]; //first two array cells are for identifying the corners, last one is top and bottom of cell
        //note, only care about about the above pairs, in 3rd index 0 is bot 1 is top

        //initialize all identifiers to -1, this is a position in the list of points that doesn't exist
        for (int di = 0; di < 4; di++)
          for (int dj = 0; dj < 4; dj++)
            intermed[di][dj][0] = intermed[di][dj][1] = -1;

        //for all considered pairs 0-1 0-2 0-3 1-2 2-3, we check to see if they span a Z-dimension boundary
        //of the cell. If they do, we add an intermediate point that stops at the cell boundaries. And we
        //update the intermediate matrix so that we know what is the index of the intermediate point.
        for (int first = 0; first < 3; first++)
          for (int second = first + 1; second < 4; second++)
            if (first != 1 || second != 3)
            {
              if (cornerPos[first] == 0)
              {
                if (cornerPos[second] < 0)
                {
                  intermed[first][second][0] = pointsInCell.size();
                  pointsInCell.push_back( Vector3<float>( getIntermediate(corners[first], corners[second], cellBot) ) );
                  edgesInCell.push_back( Edge< int >(second, pointsInCell.size() - 1));
                  edgesInCell.push_back( Edge< int >(first, pointsInCell.size() - 1));
                }
                else if (cornerPos[second] > 0)
                {
                  intermed[first][second][1] = pointsInCell.size();
                  pointsInCell.push_back( Vector3<float>( getIntermediate(corners[first], corners[second], cellTop) ) );
                  edgesInCell.push_back( Edge< int >(second, pointsInCell.size() - 1));
                  edgesInCell.push_back( Edge< int >(first, pointsInCell.size() - 1));
                }
              }
              else if (cornerPos[first] > 0)
              {

                if (cornerPos[second] == 0)
                {
                  intermed[first][second][1] = pointsInCell.size();
                  pointsInCell.push_back( Vector3<float>( getIntermediate(corners[first], corners[second], cellTop) ) );
                  edgesInCell.push_back( Edge< int >(first, pointsInCell.size() - 1));
                  edgesInCell.push_back( Edge< int >(second, pointsInCell.size() - 1));
                }
                else if (cornerPos[second] < 0)
                {
                  intermed[first][second][1] = pointsInCell.size();
                  intermed[first][second][0] = pointsInCell.size() + 1;
                  pointsInCell.push_back( Vector3<float>( getIntermediate(corners[first], corners[second], cellTop) ) );
                  pointsInCell.push_back( Vector3<float>( getIntermediate(corners[first], corners[second], cellBot) ) );
                  edgesInCell.push_back( Edge< int >(first, pointsInCell.size() - 2));
                  edgesInCell.push_back( Edge< int >(second, pointsInCell.size() - 1));
                }
              }
              else
              {
                if (cornerPos[second] == 0)
                {
                  intermed[first][second][0] = pointsInCell.size();
                  pointsInCell.push_back( Vector3<float>( getIntermediate(corners[first], corners[second], cellBot) ) );
                  edgesInCell.push_back( Edge< int >(first, pointsInCell.size() - 1));
                  edgesInCell.push_back( Edge< int >(second, pointsInCell.size() - 1));
                }
                else if (cornerPos[second] > 0)
                {
                  intermed[first][second][1] = pointsInCell.size();
                  intermed[first][second][0] = pointsInCell.size() + 1;
                  pointsInCell.push_back( Vector3<float>( getIntermediate(corners[first], corners[second], cellTop) ) );
                  pointsInCell.push_back( Vector3<float>( getIntermediate(corners[first], corners[second], cellBot) ) );
                  edgesInCell.push_back( Edge< int >(second, pointsInCell.size() - 2));
                  edgesInCell.push_back( Edge< int >(first, pointsInCell.size() - 1));
                }
              }
            }

            //if there is a top and bottom on any intermediate set, create an edge
            for (int first = 0; first < 3; first++)
              for (int second = first + 1; second < 4; second++)
                if (first != 1 || second != 3)
                  if (intermed[first][second][0] != -1 && intermed[first][second][1] != -1)
                    edgesInCell.push_back(Edge<int>(intermed[first][second][0], intermed[first][second][1]));

            //intermediates who share a corner on the same plane as them form an edge
            //unless the diagonal is also there
            for (int tier = 0; tier < 2; tier ++)
            {
              if (cornerPos[1] == -1 + (2 * tier) && intermed[0][1][tier] != -1 && intermed[1][2][tier] != -1) //-1 for bottom, 1 for top
              {
                edgesInCell.push_back(Edge<int>(intermed[0][1][tier], intermed[1][2][tier]));
              }
              if (cornerPos[3] == -1 + (2 * tier) && intermed[0][3][tier] != -1 && intermed[2][3][tier] != -1) //-1 for bottom, 1 for top
              {
                edgesInCell.push_back(Edge<int>(intermed[0][3][tier], intermed[2][3][tier]));
              }
            }


            //if the diagonal is completely in the cell create a mid a
            //and attatch to all intermediates or corners if the intermeds doesn't exist
            if (cornerPos[0] == 0 && cornerPos[2] == 0)
            {
              pointsInCell.push_back(Vector3<float>( (corners[0][0] + corners[2][0]) / 2.0f, (corners[0][1] + corners[2][1]) / 2.0f,
                                                     (corners[0][2] + corners[2][2]) / 2.0f));
              int newP = pointsInCell.size() - 1;
              edgesInCell.push_back(Edge<int>(0, newP));
              edgesInCell.push_back(Edge<int>(2, newP));
              if ( cornerPos[1] == 0)
                edgesInCell.push_back(Edge<int>(1,newP));
              else
                for (int tier = 0; tier < 2; tier++)
                {
                  if (intermed[0][1][tier] != -1)
                    edgesInCell.push_back(Edge<int>(intermed[0][1][tier], newP));
                  if (intermed[1][2][tier] != -1)
                    edgesInCell.push_back(Edge<int>(intermed[1][2][tier], newP));
                }
              if ( cornerPos[3] == 0)
                edgesInCell.push_back(Edge<int>(3,newP));
              else
                for (int tier = 0; tier < 2; tier++)
                {
                  if (intermed[0][3][tier] != -1)
                    edgesInCell.push_back(Edge<int>(intermed[0][3][tier], newP));
                  if (intermed[2][3][tier] != -1)
                    edgesInCell.push_back(Edge<int>(intermed[2][3][tier], newP));
                }
            }
            //if there is one diagonal intermed, run the same idea as above
            //note: code will be quite different. But this should connect
            //to all intermediates, and all corners if they are missing an
            //intermediate pair.---- != is essentially XOR
            else if ( (intermed[0][2][0] != -1)  != (intermed[0][2][1] != -1) )
            {
              int midP = (intermed[0][2][0] != -1 ? intermed[0][2][0] : intermed[0][2][1]);
              //only need to check 1 and 3 corners
              //since there is only one intermediate on the diagonal, either 0 or 2 exists in the cell
              //because of this, if 1 or 3 exists in the cell the intermediate always connects to it
              if ( (cornerPos[1] == -1 && (intermed[0][1][0] == -1 || intermed[1][2][0] == -1) ) ||
                   (cornerPos[1] == 1 && (intermed[0][1][1] == -1 || intermed[1][2][1] == -1) ) ||
                   cornerPos[1] == 0)
                edgesInCell.push_back(Edge<int>(1, midP));
              if ( (cornerPos[3] == -1 && (intermed[0][3][0] == -1 || intermed[2][3][0] == -1) ) ||
                   (cornerPos[3] == 1 && (intermed[0][3][1] == -1 || intermed[2][3][1] == -1) ) ||
                   cornerPos[3] == 0)
                edgesInCell.push_back(Edge<int>(3, midP));
              for (int first = 0; first < 3; first++)
                for (int second = first + 1; second < 4; second++)
                  if ( (first != 1 || second != 3) && (first != 0 || second != 2) )
                  {
                    if (intermed[first][second][0] != -1)
                      edgesInCell.push_back(Edge<int>(intermed[first][second][0], midP));
                    if (intermed[first][second][1] != -1)
                      edgesInCell.push_back(Edge<int>(intermed[first][second][1], midP));
                  }
            }
            //if there is both diagonal intermeds, connect top with all tops,
            //bot with all bots, then top to all bot intermediates
            else if (intermed[0][2][0] != -1  && intermed[0][2][1] != -1)
            {
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
                if ( (first != 1 || second != 3) && (first != 0 || second != 2) )
                {
                  if (intermed[first][second][0] != -1)
                  {
                    edgesInCell.push_back(Edge<int>(intermed[first][second][0], midB));
                    edgesInCell.push_back(Edge<int>(intermed[first][second][0], midT));
                  }
                  if (intermed[first][second][1] != -1)
                    edgesInCell.push_back(Edge<int>(intermed[first][second][1], midT));
                }
            }
            //in this case 0 and 2 are either both above or below
            //this breaks down into further cases
            else
            {
              //Note: there should be no case where all points are above or below
              //this algorithm should not execute under that circumstance

              int topBot = (cornerPos[0] > 0 ? 1 : 0); //if the diagonal is across the top, we make a mesh across the top
              //else we make a mesh across the bottom

              if (cornerPos[1] != cornerPos[0] && cornerPos[3] != cornerPos[0]) //if both corners are away from the diagonal
              { //create a mesh on the top of the cell
                edgesInCell.push_back(Edge<int>(intermed[0][3][topBot], intermed[0][1][topBot]));
                edgesInCell.push_back(Edge<int>(intermed[0][3][topBot], intermed[1][2][topBot]));
                edgesInCell.push_back(Edge<int>(intermed[0][3][topBot], intermed[2][3][topBot]));
                edgesInCell.push_back(Edge<int>(intermed[0][1][topBot], intermed[1][2][topBot]));
                edgesInCell.push_back(Edge<int>(intermed[2][3][topBot], intermed[1][2][topBot]));
                if (cornerPos[1] == (topBot == 1 ? -1 : 1)) //if the diag is on the cell top, we check if the corner is out of the bottom vice versa.
                  edgesInCell.push_back(Edge<int>(intermed[0][1][1], intermed[1][2][0])); //make the intermediate face into triangles
                if (cornerPos[3] == (topBot == 1 ? -1 : 1)) //if the diag is on the cell top, we check if the corner is out of the bottom vice versa.
                  edgesInCell.push_back(Edge<int>(intermed[0][3][1], intermed[2][3][0])); //make the intermediate face into triangles
              }
              else //at least one has to be
              {
                //triangles from up corner to opposing intermediates
                if (cornerPos[1] == cornerPos[0]) //either c1 to c3 intermeds
                {
                  edgesInCell.push_back(Edge<int>(1, intermed[0][3][topBot]));
                  edgesInCell.push_back(Edge<int>(1, intermed[2][3][topBot]));
                }
                else //or c3 to c1 intermeds
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

        cells[CELL(i,j,k)] = Cell(pointsInCell, edgesInCell, intermed,
                                            Vector3<float>(corners[0][0], corners[0][1], cellBot),
                                            Vector3<float>(corners[1][0] - corners[0][0], corners[0][1] - corners[3][1], dz_array[k]));
      }
    }
}

Vector3<float> DTEHeightField::getIntermediate(Vector3<float> a, Vector3<float> b, float height) const

{

  if (a[2] == b[2])
    return Vector3<float> ( (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, a[2]);

  float offset = a[2] - height;
  float xCoord, yCoord;
  if (a[0] != b[0])
  {
    float slopeX;
    slopeX = (b[2] - a[2]) / (b[0] - a[0]);
    float xOff = fabs((offset)/slopeX);
    xCoord = a[0] < b[0] ? a[0] + xOff : a[0] - xOff;

  }
  else
    xCoord = a[0];

  if (a[1] != b[1])
  {
    float slopeY;
    slopeY = (b[2] - a[2]) / (b[1] - a[1]);
    float yOff = fabs((offset)/slopeY);
    yCoord = a[1] < b[1] ? a[1] + yOff : a[1] - yOff;
  }
  else
    yCoord = a[1];

  return Vector3<float>(xCoord, yCoord, height);

}

void DTEHeightField::closeScanner()
{
  CPLFree(pafScanline);
}
