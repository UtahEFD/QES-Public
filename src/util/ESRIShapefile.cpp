/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
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
 * @file ESRIShapefile.cpp
 * @brief :document this:
 *
 * :long desc here:
 *
 * @sa PolyVertex
 */

#include <iostream>
#include "ESRIShapefile.h"

ESRIShapefile::ESRIShapefile()
  : minBound(2), maxBound(2), m_poDS(nullptr), m_SpRef(nullptr)
{
  minBound = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
  maxBound = { -1.0 * std::numeric_limits<float>::max(), -1.0 * std::numeric_limits<float>::max() };
}

// OBSOLETE CODE REMOVED BY FMARGAIRAZ
#if 0
ESRIShapefile::ESRIShapefile(const std::string &filename,
                             const std::string &bldLayerName,
                             std::vector<std::vector<polyVert>> &polygons,
                             std::vector<float> &building_height,
                             float heightFactor)
  : m_filename(filename), m_layerName(bldLayerName), minBound(2), maxBound(2)
{
  minBound = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
  maxBound = { -1.0 * std::numeric_limits<float>::max(), -1.0 * std::numeric_limits<float>::max() };

  GDALAllRegister();
  loadVectorData(polygons, building_height, heightFactor);
}
#endif
// OBSOLETE CODE REMOVED BY FMARGAIRAZ
#if 0
ESRIShapefile::ESRIShapefile(const std::string &filename,
                             const std::string &bldLayerName,
                             std::vector<std::vector<polyVert>> &polygons,
                             std::map<std::string, std::vector<float>> &features)
  : m_filename(filename), m_layerName(bldLayerName), minBound(2), maxBound(2)
{
  minBound = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
  maxBound = { -1.0 * std::numeric_limits<float>::max(), -1.0 * std::numeric_limits<float>::max() };

  GDALAllRegister();
  loadVectorData(polygons, features);
}
#endif
ESRIShapefile::ESRIShapefile(const std::string &filename, const std::string &bldLayerName)
  : m_filename(filename), m_layerName(bldLayerName), minBound(2), maxBound(2)
{
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "[ESRIShapefile]\t Initialization of shapefile reader..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  minBound = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
  maxBound = { -1.0 * std::numeric_limits<float>::max(), -1.0 * std::numeric_limits<float>::max() };

  GDALAllRegister();

  // From -- http://www.gdal.org/gdal_tutorial.html
  m_poDS = (GDALDataset *)GDALOpenEx(m_filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
  if (m_poDS == nullptr) {
    std::cerr << "ESRIShapefile -- could not open data file: " << m_filename << std::endl;
    exit(1);
  } else {
    std::cout << "Reading ESRIShapefile: " << m_filename << std::endl;
    printf("SHP File Driver: %s/%s\n",
           m_poDS->GetDriver()->GetDescription(),
           m_poDS->GetDriver()->GetMetadataItem(GDAL_DMD_LONGNAME));

    m_SpRef = (OGRSpatialReference *)m_poDS->GetSpatialRef();

    if (m_SpRef) {
      m_SpRef->dumpReadable();
    }

    loadVectorData(m_polygons, m_features);
  }
  std::cout << "Bounds of SHP: Min=(" << minBound[0] << ", " << minBound[1] << "), Max=(" << maxBound[0] << ", " << maxBound[1] << ")" << std::endl;
  std::cout << "Domain Size: " << (int)ceil(maxBound[0] - minBound[0]) << " X " << (int)ceil(maxBound[1] - minBound[1]) << std::endl;
  std::cout << "[ESRIShapefile]\t Loading completed." << std::endl;

  auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
  std::chrono::duration<float> elapsed_cut = finish - start;
  std::cout << "\t\t elapsed time: " << elapsed_cut.count() << " s\n";
}

// OBSOLETE CODE REMOVED BY FMARGAIRAZ
#if 0
void ESRIShapefile::loadVectorData(std::vector<std::vector<polyVert>> &polygons, std::vector<float> &building_height, float heightFactor)
{
  int polyCount = 0;

  // From -- http://www.gdal.org/gdal_tutorial.html
  m_poDS = (GDALDataset *)GDALOpenEx(m_filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
  if (m_poDS == nullptr) {
    std::cerr << "ESRIShapefile -- could not open data file: " << m_filename << std::endl;
    exit(1);
  }

  printf("SHP File Driver: %s/%s\n",
         m_poDS->GetDriver()->GetDescription(),
         m_poDS->GetDriver()->GetMetadataItem(GDAL_DMD_LONGNAME));

  // OGRLayer  *poLayer;
  // poLayer = poDS->GetLayerByName( "point" );

  int numLayers = m_poDS->GetLayerCount();
  std::cout << "Number of Layers: " << numLayers << std::endl;

  // Dump the layers for now
  for (auto i = 0; i < numLayers; i++) {
    OGRLayer *poLayer = m_poDS->GetLayer(i);
    std::cout << "\tLayer: " << poLayer->GetName() << std::endl;
  }

  // just what I set my layer name too -- may need to specify this
  //
  // Need to extract the layer from the QU XML files....
  OGRLayer *buildingLayer = m_poDS->GetLayerByName(m_layerName.c_str());
  if (buildingLayer == nullptr) {
    std::cerr << "ESRIShapefile -- no layer" << std::endl;
    exit(1);
  }

  // for all features in the building layer
  // for (auto const& poFeature: buildingLayer) {
  OGRFeature *feature = nullptr;
  OGRFeatureDefn *poFDefn = buildingLayer->GetLayerDefn();

  while ((feature = buildingLayer->GetNextFeature()) != nullptr) {

    if (poFDefn->GetFieldCount() > 1) {
      // for( auto&& oField: *feature ) {
      for (int idxField = 0; idxField < poFDefn->GetFieldCount(); idxField++) {

        OGRFieldDefn *oField = poFDefn->GetFieldDefn(idxField);
        // std::cout << "Field Name: " << oField->GetNameRef() << ", Value: ";
        if (strcmp(oField->GetNameRef(), "H") == 0) {
          switch (oField->GetType()) {
          case OFTInteger:
            // printf( "%d,", feature->GetFieldAsInteger( idxField ) );
            building_height.push_back(feature->GetFieldAsInteger(idxField) * heightFactor);
            break;
          case OFTInteger64:
            // printf( CPL_FRMT_GIB ",", feature->GetFieldAsInteger64( idxField ));
            building_height.push_back(feature->GetFieldAsInteger(idxField) * heightFactor);
            break;
          case OFTReal:
            // printf( "%.3f,", feature->GetFieldAsDouble( idxField ) );
            building_height.push_back(feature->GetFieldAsDouble(idxField) * heightFactor);
            break;
          case OFTString:
            // printf( "%s,", feature->GetFieldAsString( idxField ) );
            break;
          default:
            // printf( "%s,", feature->GetFieldAsString( idxField ) );
            break;
          }
        }
        // std::cout << std::endl;
      }
    } else {
      int idxField = 0;
      OGRFieldDefn *oField = poFDefn->GetFieldDefn(idxField);
      // std::cout << "Field Name: " << oField->GetNameRef() << ", Value: ";
      switch (oField->GetType()) {
      case OFTInteger:
        // printf( "%d,", feature->GetFieldAsInteger( idxField ) );
        building_height.push_back(feature->GetFieldAsInteger(idxField) * heightFactor);
        break;
      case OFTInteger64:
        // printf( CPL_FRMT_GIB ",", feature->GetFieldAsInteger64( idxField ));
        building_height.push_back(feature->GetFieldAsInteger(idxField) * heightFactor);
        break;
      case OFTReal:
        // printf( "%.3f,", feature->GetFieldAsDouble( idxField ) );
        building_height.push_back(feature->GetFieldAsDouble(idxField) * heightFactor);
        break;
      case OFTString:
        // printf( "%s,", feature->GetFieldAsString( idxField ) );
        break;
      default:
        // printf( "%s,", feature->GetFieldAsString( idxField ) );
        break;
      }
      // std::cout << std::endl;
    }

    OGRGeometry *poGeometry;
    poGeometry = feature->GetGeometryRef();
    if (poGeometry != nullptr
        && wkbFlatten(poGeometry->getGeometryType()) == wkbPoint) {
#if GDAL_VERSION_NUM >= GDAL_COMPUTE_VERSION(2, 3, 0)
      OGRPoint *poPoint = poGeometry->toPoint();
#else
      OGRPoint *poPoint = (OGRPoint *)poGeometry;
#endif
      printf("%.3f,%3.f\n", poPoint->getX(), poPoint->getY());
    }

    // POLYGON
    else if (poGeometry != nullptr
             && wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon) {
      // FM CLEANUP - NOT USED
      // OGRPolygon *poPolygon = (OGRPolygon *)poGeometry;

      OGRLinearRing *pLinearRing = nullptr;
      ;
      pLinearRing = ((OGRPolygon *)poGeometry)->getExteriorRing();
      int vertexCount = pLinearRing->getNumPoints();
      // std::cout << "Building Poly #" << polyCount << " (" << vertexCount << " vertices):" << std::endl;

      std::vector<polyVert> vertexList(vertexCount);

      for (int vidx = 0; vidx < vertexCount; vidx++) {
        double x = pLinearRing->getX(vidx);
        double y = pLinearRing->getY(vidx);

        if (x < minBound[0]) minBound[0] = x;
        if (y < minBound[1]) minBound[1] = y;

        if (x > maxBound[0]) maxBound[0] = x;
        if (y > maxBound[1]) maxBound[1] = y;


        // std::cout << "\t(" << x << ", " << y << ")" <<
        // std::endl;
        vertexList[vidx] = polyVert(x, y);
      }


      polyCount++;
      polygons.push_back(vertexList);


    } else {
      printf("no point geometry\n");
    }
  }

  std::cout << "Bounds of SHP: Min=(" << minBound[0] << ", " << minBound[1] << "), Max=(" << maxBound[0] << ", " << maxBound[1] << ")" << std::endl;
  std::cout << "Domain Size: " << (int)ceil(maxBound[0] - minBound[0]) << " X " << (int)ceil(maxBound[1] - minBound[1]) << std::endl;
}
#endif

void ESRIShapefile::loadVectorData(std::vector<std::vector<polyVert>> &polygons,
                                   std::map<std::string, std::vector<float>> &features)
{
  std::cout << "Summary of Vector Data:" << std::endl;
  std::cout << "----------------------------" << std::endl;
  int polyCount = 0;

  // OGRLayer  *poLayer;
  // poLayer = poDS->GetLayerByName( "point" );


  int numLayers = m_poDS->GetLayerCount();
  std::cout << "- Number of Layers: " << numLayers << std::endl;

  // Dump the layers for now
  for (auto i = 0; i < numLayers; i++) {
    OGRLayer *poLayer = m_poDS->GetLayer(i);
    std::cout << "- Layer Name: " << poLayer->GetName() << std::endl;
  }

  // just what I set my layer name too -- may need to specify this
  //
  // Need to extract the layer from the QU XML files....
  OGRLayer *buildingLayer = m_poDS->GetLayerByName(m_layerName.c_str());
  if (buildingLayer == nullptr) {
    std::cerr << "[ERROR]\t ESRIShapefile -- no layer" << std::endl;
    exit(1);
  }

  // for all features in the building layer
  // for (auto const& poFeature: buildingLayer) {
  OGRFeature *feature = nullptr;
  OGRFeatureDefn *poFDefn = buildingLayer->GetLayerDefn();

  std::cout << "- Number of Features: " << buildingLayer->GetFeatureCount() << std::endl;
  if (buildingLayer->GetFeatureCount() == 0) {
    std::cerr << "[ERROR]\t ESRIShapefile -- no feature" << std::endl;
    exit(1);
  }

  feature = buildingLayer->GetFeature(0);

  std::cout << "- Number of Fields: " << feature->GetFieldCount() << std::endl;
  if (feature->GetFieldCount() == 0) {
    std::cerr << "[ERROR]\t ESRIShapefile -- no field" << std::endl;
    exit(1);
  }

  std::cout << "- Field Names:";
  std::vector<int> validIdxField;
  for (int idxField = 0; idxField < poFDefn->GetFieldCount(); ++idxField) {
    OGRFieldDefn *oField = poFDefn->GetFieldDefn(idxField);
    std::cout << " " << oField->GetNameRef();

    switch (oField->GetType()) {
    case OFTInteger:
      validIdxField.push_back(idxField);
      break;
    case OFTInteger64:
      validIdxField.push_back(idxField);
      break;
    case OFTReal:
      validIdxField.push_back(idxField);
      break;
    case OFTString:
      // do nothing
      break;
    default:
      // do nothing
      break;
    }
  }
  std::cout << std::endl;

  while ((feature = buildingLayer->GetNextFeature()) != nullptr) {
    // for( auto&& oField: *feature ) {
    for (auto idxField : validIdxField) {

      OGRFieldDefn *oField = poFDefn->GetFieldDefn(idxField);
      std::string fn(oField->GetNameRef());
      // std::cout << "[TEST] Field Name: " << fn << std::endl;

      switch (oField->GetType()) {
      case OFTInteger:
        features[fn].push_back((float)feature->GetFieldAsInteger(idxField));
        break;
      case OFTInteger64:
        features[fn].push_back((float)feature->GetFieldAsInteger64(idxField));
        break;
      case OFTReal:
        features[fn].push_back((float)feature->GetFieldAsDouble(idxField));
        break;
      default:
        std::cerr << "[ERROR]\t ESRIShapefile -- incompatible field type" << std::endl;
        exit(1);
      }
    }
    OGRGeometry *poGeometry;
    poGeometry = feature->GetGeometryRef();
    if (poGeometry != nullptr
        && wkbFlatten(poGeometry->getGeometryType()) == wkbPoint) {
#if GDAL_VERSION_NUM >= GDAL_COMPUTE_VERSION(2, 3, 0)
      OGRPoint *poPoint = poGeometry->toPoint();
#else
      OGRPoint *poPoint = (OGRPoint *)poGeometry;
#endif
      printf("%.3f,%3.f\n", poPoint->getX(), poPoint->getY());
    }

    // POLYGON
    else if (poGeometry != nullptr
             && wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon) {
      // FM CLEANUP - NOT USED
      // OGRPolygon *poPolygon = (OGRPolygon *)poGeometry;

      OGRLinearRing *pLinearRing = nullptr;

      pLinearRing = ((OGRPolygon *)poGeometry)->getExteriorRing();
      int vertexCount = pLinearRing->getNumPoints();
      // std::cout << "Building Poly #" << polyCount << " (" << vertexCount << " vertices):" << std::endl;

      std::vector<polyVert> vertexList(vertexCount);

      for (int vidx = 0; vidx < vertexCount; vidx++) {
        auto x = (float)pLinearRing->getX(vidx);
        auto y = (float)pLinearRing->getY(vidx);

        if (x < minBound[0]) minBound[0] = x;
        if (y < minBound[1]) minBound[1] = y;

        if (x > maxBound[0]) maxBound[0] = x;
        if (y > maxBound[1]) maxBound[1] = y;


        // std::cout << "\t(" << x << ", " << y << ")" <<
        // std::endl;
        vertexList[vidx] = polyVert(x, y);
      }


      polyCount++;
      polygons.push_back(vertexList);


    } else {
      printf("no point geometry\n");
    }
  }
  std::cout << "----------------------------" << std::endl;
  // std::cout << "[TEST] number of polygons " << polygons.size() << std::endl;
}
