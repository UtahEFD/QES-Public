#include <iostream>
#include "ESRIShapefile.h"

ESRIShapefile::ESRIShapefile()
{
}

ESRIShapefile::ESRIShapefile(const std::string &filename)
    : m_filename(filename)
{
    GDALAllRegister();
    loadVectorData();
}

void ESRIShapefile::loadVectorData()
{
    // From -- http://www.gdal.org/gdal_tutorial.html
    m_poDS = (GDALDataset*) GDALOpenEx( m_filename.c_str(),
                                        GDAL_OF_VECTOR,
                                        NULL, NULL, NULL );
    if ( m_poDS == nullptr ) {
        std::cerr << "ESRIShapefile -- could not open data file: " << m_filename << std::endl;
        exit( 1 );
    }

    printf( "SHP File Driver: %s/%s\n",
            m_poDS->GetDriver()->GetDescription(), 
            m_poDS->GetDriver()->GetMetadataItem( GDAL_DMD_LONGNAME ) );

    //OGRLayer  *poLayer;
    //poLayer = poDS->GetLayerByName( "point" );

    int numLayers = m_poDS->GetLayerCount();
    std::cout << "Number of Layers: " << numLayers << std::endl;

    // Dump the layers for now
    for (auto i=0; i<numLayers; i++) {
        OGRLayer *poLayer = m_poDS->GetLayer( i );
        std::cout << "\tLayer: " << poLayer->GetName() << std::endl;
    }
    
    // just what I set my layer name too -- may need to specify this
    OGRLayer  *buildingLayer = m_poDS->GetLayerByName( "DLH_buildings" );  
    if (buildingLayer == nullptr) {
        std::cerr << "ESRIShapefile -- no layer" << std::endl;
        exit( 1 );
    }
    
    // for all features in the building layer
    //for (auto const& poFeature: buildingLayer) {
    OGRFeature* feature = nullptr;
    OGRFeatureDefn *poFDefn = buildingLayer->GetLayerDefn();

    while((feature = buildingLayer->GetNextFeature()) != nullptr) {

        // for( auto&& oField: *feature ) {
        for(int idxField = 0; idxField < poFDefn->GetFieldCount(); idxField++ ) {

            OGRFieldDefn *oField = poFDefn->GetFieldDefn( idxField );            
            std::cout << "Field Name: " << oField->GetNameRef() << ", Value: ";
            switch( oField->GetType() )
            {
            case OFTInteger:
                printf( "%d,", feature->GetFieldAsInteger( idxField ) );
                break;
            case OFTInteger64:
                printf( CPL_FRMT_GIB ",", feature->GetFieldAsInteger64( idxField ) );
                break;
            case OFTReal:
                printf( "%.3f,", feature->GetFieldAsDouble( idxField ) );
                break;
            case OFTString:
                printf( "%s,", feature->GetFieldAsString( idxField ) );
                break;
            default:
                printf( "%s,", feature->GetFieldAsString( idxField ) );
                break;
            }
            std::cout << std::endl;

            OGRGeometry *poGeometry;
            poGeometry = feature->GetGeometryRef();
            if( poGeometry != NULL
                && wkbFlatten(poGeometry->getGeometryType()) == wkbPoint )
            {
#if GDAL_VERSION_NUM >= GDAL_COMPUTE_VERSION(2,3,0)
                OGRPoint *poPoint = poGeometry->toPoint();
#else
                OGRPoint *poPoint = (OGRPoint *) poGeometry;
#endif
                printf( "%.3f,%3.f\n", poPoint->getX(), poPoint->getY() );
            }
            else if( poGeometry != NULL
                     && wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon )
            {
#if GDAL_VERSION_NUM >= GDAL_COMPUTE_VERSION(2,3,0)
                OGRPolygon *poPolygon = poGeometry->toPoint();
#else
                OGRPolygon *poPolygon = (OGRPolygon *) poGeometry;
#endif

            
                OGRLinearRing* pLinearRing = nullptr;;
                pLinearRing = ((OGRPolygon*)poGeometry)->getExteriorRing();
                int vertexCount = pLinearRing->getNumPoints();
                std::cout << "Building Poly (" << vertexCount << " vertices):" << std::endl;
                for (int vidx=0; vidx< vertexCount; vidx++) {
                    double x = pLinearRing->getX( vidx );
                    double y = pLinearRing->getY( vidx );
                    std::cout << "\t(" << x << ", " << y << ")" << std::endl;
                }
            }
            else
            {
                printf( "no point geometry\n" );
            }
        }
        
    }
}



