#include "DTEHeightField.h"

DTEHeightField::DTEHeightField()
  : m_filename(""), m_rbMin(0.0)
{
}

DTEHeightField::DTEHeightField(const std::string &filename, Shader *s)
  : m_filename(filename), m_baseShader(s), m_rbMin(0.0)
{
  GDALAllRegister();

  loadImage();

  load();
}

void DTEHeightField::loadImage()
{
  // High res giant image
  // std::string filename = "/scratch/PSP_003910_1685_RED_A_01_ORTHO.JP2";

  std::string filename = "/tmp/PSP_003910_1685_RED_C_01_ORTHO.JP2";

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

  double xGeo, yGeo;
  sivelab::Vector3D tc0, tc1, tc2;

  int step = 1;
  for (int Yline = 0; Yline < m_nYSize-1; Yline+=step) {
    for (int Xpixel = 0; Xpixel < m_nXSize-1; Xpixel+=step) {

      // For these purposes, pixel refers to the "X" coordinate, while
      // line refers to the "Z" coordinate

      xGeo = m_geoTransform[0] + Xpixel * m_geoTransform[1];
      yGeo = m_geoTransform[3] + Yline * m_geoTransform[5];
      sivelab::Vector3D tv0(Yline, queryHeight(pafScanline, Yline, Xpixel), Xpixel );
      convertToTexCoord(Yline, Xpixel, tc0);

      sivelab::Vector3D tv1(Yline, queryHeight(pafScanline, Yline, Xpixel+step), Xpixel+step );
      convertToTexCoord(Yline, Xpixel+step, tc1);

      xGeo = m_geoTransform[0] + Xpixel * m_geoTransform[1];
      yGeo = m_geoTransform[3] + Yline * m_geoTransform[5];
      sivelab::Vector3D tv2(Yline+step, queryHeight(pafScanline, Yline+step, Xpixel), Xpixel);
      convertToTexCoord(Yline+step, Xpixel, tc2);

      // Pull texture coordinates and store here...  
      // 
      // - find the the georeferenced coordinate for these particle
      //   vertices, and look up the equivalent index in the image data

      tPtr = new Triangle( tv0, tv1, tv2 );
      tPtr->setTextureCoordinates( tc0, tc1, tc2 );
      tPtr->provideShader( m_baseShader );
      
      m_triList.push_back(tPtr);
      
      m_bbox.update(tv0);
      m_bbox.update(tv1);
      m_bbox.update(tv2);

      sivelab::Vector3D tv3( Yline+step, queryHeight( pafScanline, Yline+step, Xpixel ), Xpixel );
      convertToTexCoord(Yline+step, Xpixel, tc0);

      sivelab::Vector3D tv4( Yline,   queryHeight( pafScanline, Yline, Xpixel+step ), Xpixel+step );
      convertToTexCoord(Yline, Xpixel+step, tc1);

      sivelab::Vector3D tv5( Yline+step, queryHeight( pafScanline, Yline+step, Xpixel+step ), Xpixel+step );
      convertToTexCoord(Yline+step, Xpixel+step, tc2);

      tPtr = new Triangle( tv3, tv4, tv5 );
      tPtr->setTextureCoordinates( tc0, tc1, tc2 );
      tPtr->provideShader( m_baseShader );
      
      m_triList.push_back(tPtr);
      
      m_bbox.update(tv3);
      m_bbox.update(tv4);
      m_bbox.update(tv5);
    }
  }

  CPLFree(pafScanline);

  std::cout << "Generating BVH with " << m_triList.size() << " triangles." << std::endl;

  m_modelRoot = new BoundingVolumeNode( m_triList, 0 );

  std::cout << "m_bbox = " << m_bbox << std::endl;
}

bool DTEHeightField::closestHit(const Ray &r, const double t0, double &t1, HitStruct &hit, bool primRay)
{
  if (m_modelRoot->closestHit(r, t0, t1, hit, primRay))
    return true;
  else 
    return false;
}


DTEHeightField::~DTEHeightField()
{
  GDALClose(m_poDataset);
}

