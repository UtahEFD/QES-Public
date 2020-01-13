#include "DTEHeightField.h"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
#define LIMIT 99999999.0f

DTEHeightField::DTEHeightField()
  : m_filename(""), m_rbMin(0.0)
{
  m_poDataset = 0;

}

DTEHeightField::DTEHeightField(const std::string &filename, double cellSizeXN, double cellSizeYN)
  : m_filename(filename), m_rbMin(0.0), cellSizeX(cellSizeXN), cellSizeY(cellSizeYN)
{
  GDALAllRegister();

  // only for texture???
 // loadImage();

  load();
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

  // From -- http://www.gdal.org/gdal_tutorial.html
  m_poDataset = (GDALDataset *) GDALOpen( m_filename.c_str(), GA_ReadOnly );
  if( m_poDataset == NULL )
    {
      std::cerr << "Couldn't open file: " << m_filename << std::endl;
      exit( EXIT_FAILURE );
    }

  printf( "Driver: %s/%s\n",
	  m_poDataset->GetDriver()->GetDescription(),
	  m_poDataset->GetDriver()->GetMetadataItem( GDAL_DMD_LONGNAME ) );

  printf( "Size is %dx%dx%d\n",
	  m_poDataset->GetRasterXSize(), m_poDataset->GetRasterYSize(),
	  m_poDataset->GetRasterCount() );

  if( m_poDataset->GetProjectionRef()  != NULL )
    printf( "Projection is `%s'\n", m_poDataset->GetProjectionRef() );

  if( m_poDataset->GetGeoTransform( m_geoTransform ) == CE_None )
    {
      printf( "Origin = (%.6f,%.6f)\n",
	      m_geoTransform[0], m_geoTransform[3] );

      printf( "Pixel Size = (%.6f,%.6f)\n",
	      m_geoTransform[1], m_geoTransform[5] );
      pixelSizeX = abs(m_geoTransform[1]);
      pixelSizeY = abs(m_geoTransform[5]);

      printf( "These should be zero for north up: (%.6f, %.6f)\n",
	      m_geoTransform[2], m_geoTransform[4] );
    }

  GDALRasterBand  *poBand;
  int             nBlockXSize, nBlockYSize;
  int             bGotMin, bGotMax;
  double          adfMinMax[2];

  poBand = m_poDataset->GetRasterBand( 1 );
  poBand->GetBlockSize( &nBlockXSize, &nBlockYSize );
  printf( "Block=%dx%d Type=%s, ColorInterp=%s\n",
	  nBlockXSize, nBlockYSize,
	  GDALGetDataTypeName(poBand->GetRasterDataType()),
	  GDALGetColorInterpretationName(poBand->GetColorInterpretation()) );

  adfMinMax[0] = poBand->GetMinimum( &bGotMin );
  adfMinMax[1] = poBand->GetMaximum( &bGotMax );
  if( ! (bGotMin && bGotMax) )
    GDALComputeRasterMinMax((GDALRasterBandH)poBand, TRUE, adfMinMax);

  printf( "Min=%.3fd, Max=%.3f\n", adfMinMax[0], adfMinMax[1] );

  if( poBand->GetOverviewCount() > 0 )
    printf( "Band has %d overviews.\n", poBand->GetOverviewCount() );

  if( poBand->GetColorTable() != NULL )
    printf( "Band has a color table with %d entries.\n",
	    poBand->GetColorTable()->GetColorEntryCount() );

  m_rbScale = poBand->GetScale();
  printf( "Band has scale: %.4f\n", m_rbScale );

  m_rbOffset = poBand->GetOffset();
  printf( "Band has offset: %.4f\n", m_rbOffset );

  m_rbNoData = poBand->GetNoDataValue();
  printf( "Band has NoData value: %.4f\n", m_rbNoData );


  m_nXSize = poBand->GetXSize();
  m_nYSize = poBand->GetYSize();

  printf( "DEM size is %dx%dx%d\n",
	  m_nXSize, m_nYSize );

  pafScanline = (float *) CPLMalloc(sizeof(float)*m_nXSize*m_nYSize);

  // rb->RasterIO(GF_Read, 0, 0, xsize, ysize, buffer, xsize, ysize, GDT_Float32, 0, 0);
  //
  // CPLErr - CE_Failure if the access fails, otherwise CE_None.
  CPLErr rasterErr = poBand->RasterIO( GF_Read, 0, 0,
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

  float stepX = cellSizeX / pixelSizeX; // tie back to dx, dy here.... with scaling of pixelsize
  float stepY = cellSizeY / pixelSizeY;

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
    printf("domain: %d %d %d\n", (*domain)[0], (*domain)[1], (*domain)[2]);
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
           std::cout << "height:  " << queryHeight( pafScanline , ( dx) / pixelSizeX,  ( dy) / pixelSizeY) - min[2] << std::endl;
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

       if ( i == 256 && j == 41)
       {
         std::cout<< "corners[0]:  " << corners[0][2] << std::endl;
         std::cout<< "corners[1]:  " << corners[1][2] << std::endl;
         std::cout<< "corners[2]:  " << corners[2][2] << std::endl;
         std::cout<< "corners[3]:  " << corners[3][2] << std::endl;
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

    if ( cellTop < coordsMin)
      cells[CELL(i,j,k)] = Cell(terrain_CT, Vector3<float>(corners[0][0], corners[0][1], cellBot),
                                            Vector3<float>(corners[1][0] - corners[0][0], corners[0][1] - corners[3][1], dz_array[k]));
    else if ( cellBot > coordsMax)
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
