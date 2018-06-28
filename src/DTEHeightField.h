#ifndef __DTE_HEIGHT_FIELD_H__
#define __DTE_HEIGHT_FIELD_H__ 1

// Copyright Pete Willemsen
// MIT

#include <string>

#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()

class DTEHeightField
{
public:
  DTEHeightField();
  DTEHeightField(const std::string &filename);
  ~DTEHeightField();

  bool closestHit(const Ray &ray, const double t0, double &t1, HitStruct &hit, bool primRay=true);

private:
  void load();
  void loadImage();

  bool compareEquality( double f1, double f2 )
  {
    const double eps = 1.0e-6;
    return fabs( f1 - f2 ) < eps;
  }

  float queryHeight( float *scanline, int j, int k )
  {
    double height = scanline[ j * m_nXSize + k ];
    if (!compareEquality( height, m_rbNoData ))
      height = height * m_rbScale + m_rbOffset;
    else 
      height = m_rbMin;
    return height;
  }

  void convertToTexCoord(int j, int k, sivelab::Vector3D &texCoord)
  {
    double xGeo = m_geoTransform[0] + k * m_geoTransform[1];
    double yGeo = m_geoTransform[3] + j * m_geoTransform[5];

    int image_i = (xGeo - m_imageGeoTransform[0]) / m_imageGeoTransform[1];
    int image_j = (yGeo - m_imageGeoTransform[3]) / m_imageGeoTransform[5];

    if (image_i < 0) image_i = 0;
    if (image_i > m_imageXSize) image_i = m_imageXSize - 1;

    if (image_j < 0) image_j = 0;
    if (image_j > m_imageYSize) image_j = m_imageYSize - 1;

    // std::cout << "GeoRef = (" << xGeo << ", " << yGeo << "), image coord = (" << image_i << ", " << image_j << "), (s,t) = (" << image_i / (float)m_imageXSize << ", " << image_j / (float)m_imageYSize << ")" << std::endl;

    texCoord[0] = image_i / (float)m_imageXSize;
    texCoord[1] = image_j / (float)m_imageYSize;
    texCoord[2] = 0.0;
  }

  std::string m_filename;
  GDALDataset  *m_poDataset;
  double m_geoTransform[6];  

  int m_nXSize, m_nYSize;
  double m_rbScale, m_rbOffset, m_rbNoData, m_rbMin;

  // Texture relative information
  GDALDataset  *m_imageDataset;
  int m_imageXSize, m_imageYSize;
  double m_imageGeoTransform[6];  
  
  BoundingVolumeNode *m_modelRoot;
  std::vector<RenderObject*> m_triList;
};


#endif
