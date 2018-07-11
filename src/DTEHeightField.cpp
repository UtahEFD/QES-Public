#include "DTEHeightField.h"

DTEHeightField::DTEHeightField()
  : m_filename(""), m_rbMin(0.0)
{
}

DTEHeightField::DTEHeightField(const std::string &filename)
  : m_filename(filename), m_rbMin(0.0)
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

  int step = 80;
  for (int Yline = 0; Yline < m_nYSize-1; Yline+=step) {
    for (int Xpixel = 0; Xpixel < m_nXSize-1; Xpixel+=step) {

      int iXStep = step;
      int iYStep = step;
      if ( (step + Yline) > m_nYSize)
        iYStep = m_nYSize - 1 - Yline;
      if ( (step + Xpixel) > m_nXSize)
        iXStep = m_nXSize - 1 - Xpixel;

      // For these purposes, pixel refers to the "X" coordinate, while
      // line refers to the "Z" coordinate

      // turn localized coordinates (Yline and Xpixel) into geo-referenced values.
      // then use the geo referenced coordinate to lookup the height.


//      xGeo = m_geoTransform[0] + Xpixel * m_geoTransform[1];
//      yGeo = m_geoTransform[3] + Yline * m_geoTransform[5];

//height set to y coord, change to z

      Vector3<float> tv0((float)Yline, (float)Xpixel, queryHeight( pafScanline, Yline, Xpixel) );

      Vector3<float> tv1((float)Yline, (float)(Xpixel + iXStep), queryHeight( pafScanline, Yline, Xpixel + iXStep) );

//      xGeo = m_geoTransform[0] + Xpixel * m_geoTransform[1];
//      yGeo = m_geoTransform[3] + Yline * m_geoTransform[5];
      Vector3<float> tv2((float)Yline + iYStep, (float)Xpixel, queryHeight( pafScanline, Yline + iYStep, Xpixel));

      tPtr = new Triangle( tv0, tv1, tv2 );
      m_triList.push_back(tPtr);

      Vector3<float> tv3( (float)(Yline + iYStep), (float)Xpixel, queryHeight( pafScanline, Yline + iYStep, Xpixel ) );
     // convertToTexCoord(Yline+step, Xpixel, tc0);

      Vector3<float> tv4( (float)Yline, (float)(Xpixel + iXStep),   queryHeight( pafScanline, Yline, Xpixel + iXStep ) );
      // convertToTexCoord(Yline, Xpixel+step, tc1);

      Vector3<float> tv5( (float)(Yline + iYStep), (float)(Xpixel + iXStep), queryHeight( pafScanline, Yline+ iYStep, Xpixel + iXStep ) );
      // convertToTexCoord(Yline+step, Xpixel+step, tc2);

      tPtr = new Triangle( tv3, tv4, tv5 );
      m_triList.push_back(tPtr);
    }
    std::cout << "DEM Loading - percentage complete: " << Yline << ", " << (Yline / (float)m_nYSize) * 100.0f << std::endl;
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
    float min[3] = {999999999.0f, 999999999.0f, 999999999.0f};
    float max[3] = {0.0f, 0.0f, 0.0f};
    for (int q = 0; q < 3; q++)
    {
      for (int i = 0; i < m_triList.size(); i++)
      {
        if ( (*(m_triList[i]->a))[q] > 0 && (*(m_triList[i]->a))[q] < min[q] )
          min[q] = (*(m_triList[i]->a))[q];
        if ( (*(m_triList[i]->b))[q] > 0 && (*(m_triList[i]->b))[q] < min[q] )
          min[q] = (*(m_triList[i]->b))[q];
        if ( (*(m_triList[i]->c))[q] > 0 && (*(m_triList[i]->c))[q] < min[q] )
          min[q] = (*(m_triList[i]->c))[q];

        if ( (*(m_triList[i]->a))[q] > max[q] )
          max[q] = (*(m_triList[i]->a))[q];
        if ( (*(m_triList[i]->b))[q] > max[q] )
          max[q] = (*(m_triList[i]->b))[q];
        if ( (*(m_triList[i]->c))[q] > max[q] )
          max[q] = (*(m_triList[i]->c))[q];
      }

      for (int i = 0; i < m_triList.size(); i++)
      {
        (*(m_triList[i]->a))[q] -= min[q];
        (*(m_triList[i]->b))[q] -= min[q];
        (*(m_triList[i]->c))[q] -= min[q];
      }

      max[q] -= min[q];
      (*domain)[q] = (int)(max[q] / (*grid)[q]) + 1;

      //current implementation adds buffer in z dim for buffer space
      //get more specific values, currently adding 50 meters
      //Also, domains are currently only working with cubic dimensions... fix this
      
      if (q == 2)
        (*domain)[q] += (int)(50.0f / (*grid)[q]);      

    }
    if ((*domain)[0] >= (*domain)[1] && (*domain)[0] >= (*domain)[2])
      (*domain)[1] = (*domain) [2] = (*domain)[0];
    else if ((*domain)[1] >= (*domain)[0] && (*domain)[1] >= (*domain)[2])
      (*domain)[0] = (*domain) [2] = (*domain)[1];
    else
        (*domain)[0] = (*domain) [1] = (*domain)[2];
    printf("domain: %d %d %d\n", (*domain)[0], (*domain)[1], (*domain)[2]);
}