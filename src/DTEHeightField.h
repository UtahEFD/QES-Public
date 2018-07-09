#ifndef __DTE_HEIGHT_FIELD_H__
#define __DTE_HEIGHT_FIELD_H__ 1

// Copyright Pete Willemsen
// MIT

#include <string>
#include "Triangle.h"
#include "Vector3.h"
#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()

class DTEHeightField
{
public:
  DTEHeightField();
  DTEHeightField(const std::string &filename);
  ~DTEHeightField();

  std::vector<Triangle*> getTris() {return m_triList;}


private:
  void load();

  // void loadImage();

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


  std::string m_filename;
  GDALDataset  *m_poDataset;
  double m_geoTransform[6];  

  int m_nXSize, m_nYSize;
  double m_rbScale, m_rbOffset, m_rbNoData, m_rbMin;

  // Texture relative information
  GDALDataset  *m_imageDataset;
  int m_imageXSize, m_imageYSize;
  double m_imageGeoTransform[6];  
  
  std::vector<Triangle*> m_triList;
};


#endif
