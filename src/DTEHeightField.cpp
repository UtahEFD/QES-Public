#include "DTEHeightField.h"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
#define LIMIT 99999999.0f

DTEHeightField::DTEHeightField()
  : m_filename(""), m_rbMin(0.0)
{
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

  float *pafScanline;
  
  m_nXSize = poBand->GetXSize();
  m_nYSize = poBand->GetYSize();
  
  pafScanline = (float *) CPLMalloc(sizeof(float)*m_nXSize*m_nYSize);

  // rb->RasterIO(GF_Read, 0, 0, xsize, ysize, buffer, xsize, ysize, GDT_Float32, 0, 0);
  poBand->RasterIO( GF_Read, 0, 0, 
		    m_nXSize, m_nYSize, 
		    pafScanline, 
		    m_nXSize, m_nYSize, GDT_Float32, 
		    0, 0 );

  Triangle *tPtr=0;
  m_triList.clear();

  //double xGeo, yGeo;
  Vector3<float> tc0, tc1, tc2;


  std::cout << "DEM Loading\n";

  float stepX = cellSizeX / pixelSizeX; // tie back to dx, dy here.... with scaling of pixelsize
  float stepY = cellSizeY / pixelSizeY;
  if (stepX < 1) stepX = 1.0f;
  if (stepY < 1) stepY = 1.0f;
  for (float iYline = 0; iYline < m_nYSize-1; iYline+=stepY) {
    for (float iXpixel = 0; iXpixel < m_nXSize-1; iXpixel+=stepX) {

      int Yline = iYline;
      int Xpixel = iXpixel;
      int iXStep = stepX;
      int iYStep = stepY;
      if ( (stepY + Yline) > m_nYSize)
        iYStep = m_nYSize - 1 - Yline;
      if ( (stepX + Xpixel) > m_nXSize)
        iXStep = m_nXSize - 1 - Xpixel;

      // For these purposes, pixel refers to the "X" coordinate, while
      // line refers to the "Z" coordinate

      // turn localized coordinates (Yline and Xpixel) into geo-referenced values.
      // then use the geo referenced coordinate to lookup the height.


//      xGeo = m_geoTransform[0] + Xpixel * m_geoTransform[1];
//      yGeo = m_geoTransform[3] + Yline * m_geoTransform[5];

//height set to y coord, change to z

      Vector3<float> tv0((float)Yline * pixelSizeY, (float)Xpixel * pixelSizeX, queryHeight( pafScanline, Yline * pixelSizeY, Xpixel * pixelSizeX) );

      Vector3<float> tv1((float)Yline * pixelSizeY, (float)(Xpixel + iXStep) * pixelSizeX, queryHeight( pafScanline, Yline * pixelSizeY, (Xpixel + iXStep) * pixelSizeX ) );

//      xGeo = m_geoTransform[0] + Xpixel * m_geoTransform[1];
//      yGeo = m_geoTransform[3] + Yline * m_geoTransform[5];
      Vector3<float> tv2((float)(Yline + iYStep) * pixelSizeY, (float)Xpixel * pixelSizeX, queryHeight( pafScanline, (Yline + iYStep) * pixelSizeY, Xpixel * pixelSizeX));

      tPtr = new Triangle( tv0, tv1, tv2 );
      m_triList.push_back(tPtr);

      Vector3<float> tv3( (float)(Yline + iYStep) * pixelSizeY, (float)Xpixel * pixelSizeX, queryHeight( pafScanline, (Yline + iYStep) * pixelSizeY, Xpixel * pixelSizeX ) );
     // convertToTexCoord(Yline+step, Xpixel, tc0);

      Vector3<float> tv4( (float)Yline * pixelSizeY, (float)(Xpixel + iXStep) * pixelSizeX,   queryHeight( pafScanline, Yline * pixelSizeY, (Xpixel + iXStep) * pixelSizeX ) );
      // convertToTexCoord(Yline, Xpixel+step, tc1);

      Vector3<float> tv5( (float)(Yline + iYStep) * pixelSizeY, (float)(Xpixel + iXStep) * pixelSizeX, queryHeight( pafScanline, (Yline+ iYStep) * pixelSizeY, (Xpixel + iXStep) * pixelSizeX ) );
      // convertToTexCoord(Yline+step, Xpixel+step, tc2);

      tPtr = new Triangle( tv3, tv4, tv5 );
      m_triList.push_back(tPtr);
    }
    printProgress((iYline / (float)m_nYSize));
  }
  std::cout << std::endl;

  // At end of loop above, all height field data will have been converted to a triangle mesh, stored in m_triList.

  CPLFree(pafScanline);
}

DTEHeightField::~DTEHeightField()
{
  GDALClose(m_poDataset);
}

void DTEHeightField::setDomain(Vector3<int>* domain, Vector3<float>* grid)
{
    float min[3] = {LIMIT, LIMIT, LIMIT};
    float max[3] = {0.0f, 0.0f, 0.0f};
    std::cout << "Seting Terrain Boundaries\n";
    for (int q = 0; q < 3; q++)
    {
      if (q == 0)
        std::cout << "X\n";
      else if (q == 1)
        std::cout << "Y\n";
      else
        std::cout << "Z\n";

      for (int i = 0; i < m_triList.size(); i++)
      {
        if ( (*(m_triList[i]->a))[q] > 0 && (*(m_triList[i]->a))[q] < min[q] )
          min[q] = (*(m_triList[i]->a))[q];
        if ( (*(m_triList[i]->b))[q] > 0 && (*(m_triList[i]->b))[q] < min[q] )
          min[q] = (*(m_triList[i]->b))[q];
        if ( (*(m_triList[i]->c))[q] > 0 && (*(m_triList[i]->c))[q] < min[q] )
          min[q] = (*(m_triList[i]->c))[q];

        if ( (*(m_triList[i]->a))[q] > max[q] && (*(m_triList[i]->a))[q] < LIMIT)
          max[q] = (*(m_triList[i]->a))[q];
        if ( (*(m_triList[i]->b))[q] > max[q] && (*(m_triList[i]->b))[q] < LIMIT)
          max[q] = (*(m_triList[i]->b))[q];
        if ( (*(m_triList[i]->c))[q] > max[q] && (*(m_triList[i]->c))[q] < LIMIT)
          max[q] = (*(m_triList[i]->c))[q];
         printProgress( ( (float)i / (float)m_triList.size()) / 2.0f );
      }

      for (int i = 0; i < m_triList.size(); i++)
      {
        (*(m_triList[i]->a))[q] -= min[q];
        (*(m_triList[i]->b))[q] -= min[q];
        (*(m_triList[i]->c))[q] -= min[q];
        printProgress( ( (float)i / (float)m_triList.size()) / 2.0f + 0.5f); 
      }

      max[q] -= min[q];
      (*domain)[q] = (int)(max[q] / (float)(*grid)[q]) + 1;
      printf ("max %lf grid %lf\n" , max[q], (float)(*grid)[q]);

      //current implementation adds buffer in z dim for buffer space
      //get more specific values, currently adding 50 meters
      //Also, domains are currently only working with cubic dimensions... fix this
      
      if (q == 2)
        (*domain)[q] += (int)(50.0f / (float)(*grid)[q]);      

      std::cout << std::endl;
    }
    /*if ((*domain)[0] >= (*domain)[1] && (*domain)[0] >= (*domain)[2])
      (*domain)[1] = (*domain) [2] = (*domain)[0];
    else if ((*domain)[1] >= (*domain)[0] && (*domain)[1] >= (*domain)[2])
      (*domain)[0] = (*domain) [2] = (*domain)[1];
    else
        (*domain)[0] = (*domain) [1] = (*domain)[2];//*/
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