#include "LocalMixingOptix.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

void LocalMixingOptix::defineMixingLength(const WINDSInputData* WID,WINDSGeneralData* WGD) 
{

    int nx = WGD->nx;
    int ny = WGD->ny;
    int nz = WGD->nz;

    float dx = WGD->dx;
    float dy = WGD->dy;
    float dz = WGD->dz;
   
    // ///////////////////////////////////////
    //
    // Mixing Length Ray Tracing Geometry Processing
    //
    // If we're calculating mixing length, make sure to add the
    // buildings to the bvh used to calculate length scale
    
    std::vector< Triangle* > buildingTris;

    if (WGD->allBuildingsV.size() > 0) {
       
        std::cout << "Preparing building geometry data for mixing length calculations." << std::endl;

        // for (auto bIdx = 0u; bIdx <
        // WID->buildings->buildings.size(); bIdx++)
        for (auto bIdx = 0u; bIdx < WGD->allBuildingsV.size(); bIdx++)
        {
            // for each polyvert edge, create triangles of the sides
            for (auto pIdx=0u; pIdx < WGD->allBuildingsV[bIdx]->polygonVertices.size(); pIdx++)
            {
                // each (x_poly, y_poly) represents 1 vertices of the
                // polygon that is 2D.


                // Form a line between (x_poly_i, y_poly_i) and (x_poly_i+1, y_poly_i+1)
                //
                // That line can be extruded to form a plane that could
                // be decomposed into two triangles.
                //
                // Building has base_height -- should be the "ground"
                // of the building.
                //
                // Building also has height_eff, which is the height of
                // the building
                //
                // So triangle1 of face_i should be
                // (x_poly_i, y_poly_i, base_height)
                // (x_poly_i+1, y_poly_i+1, base_height)
                // (x_poly_i+1, y_poly_i+1, base_height + height_eff)
                //
                // Then triangle2 of face_i should be
                // (x_poly_i, y_poly_i, base_height)
                // (x_poly_i+1, y_poly_i+1, base_height + height_eff)
                // (x_poly_i, y_poly_i, base_height + height_eff)
                //

                if(pIdx == WGD->allBuildingsV[bIdx]->polygonVertices.size()-1){ //wrap around case for last vertices

                    //Triangle 1
                    Triangle *tri1 = new Triangle(Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[pIdx].x_poly,
                                                                 WGD->allBuildingsV[bIdx]->polygonVertices[pIdx].y_poly,
                                                                 WGD->allBuildingsV[bIdx]->base_height),
                                                  Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[0].x_poly,
                                                                 WGD->allBuildingsV[bIdx]->polygonVertices[0].y_poly,
                                                                 WGD->allBuildingsV[bIdx]->base_height),
                                                  Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[0].x_poly,
                                                                 WGD->allBuildingsV[bIdx]->polygonVertices[0].y_poly,
                                                                 WGD->allBuildingsV[bIdx]->base_height+WGD->allBuildingsV[bIdx]->H)
                                                  );

                    //Triangle 2
                    Triangle *tri2 = new Triangle(Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[pIdx].x_poly,
                                                                 WGD->allBuildingsV[bIdx]->polygonVertices[pIdx].y_poly,
                                                                 WGD->allBuildingsV[bIdx]->base_height+WGD->allBuildingsV[bIdx]->H),
                                                  Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[0].x_poly,
                                                                 WGD->allBuildingsV[bIdx]->polygonVertices[0].y_poly,
                                                                 WGD->allBuildingsV[bIdx]->base_height+WGD->allBuildingsV[bIdx]->H),
                                                  Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[0].x_poly,
                                                                 WGD->allBuildingsV[bIdx]->polygonVertices[0].y_poly,
                                                                 WGD->allBuildingsV[bIdx]->base_height)
                                                  );

                    buildingTris.push_back(tri1);
                    buildingTris.push_back(tri2);

                }else{

                    //Triangle 1
                    Triangle *tri1 = new Triangle(Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[pIdx].x_poly,
                                                                 WGD->allBuildingsV[bIdx]->polygonVertices[pIdx].y_poly,
                                                                 WGD->allBuildingsV[bIdx]->base_height),
                                                  Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[pIdx+1].x_poly,
                                                                 WGD->allBuildingsV[bIdx]->polygonVertices[pIdx+1].y_poly,
                                                                 WGD->allBuildingsV[bIdx]->base_height),
                                                  Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[pIdx+1].x_poly,
                                                                 WGD->allBuildingsV[bIdx]->polygonVertices[pIdx+1].y_poly,
                                                                 WGD->allBuildingsV[bIdx]->base_height+WGD->allBuildingsV[bIdx]->H)
                                                  );

                    //Triangle 2
                    Triangle *tri2 = new Triangle(Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[pIdx].x_poly,
                                                                 WGD->allBuildingsV[bIdx]->polygonVertices[pIdx].y_poly,
                                                                 WGD->allBuildingsV[bIdx]->base_height+WGD->allBuildingsV[bIdx]->H),
                                                  Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[pIdx+1].x_poly,
                                                                 WGD->allBuildingsV[bIdx]->polygonVertices[pIdx+1].y_poly,
                                                                 WGD->allBuildingsV[bIdx]->base_height+WGD->allBuildingsV[bIdx]->H),
                                                  Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[pIdx+1].x_poly,
                                                                 WGD->allBuildingsV[bIdx]->polygonVertices[pIdx+1].y_poly,
                                                                 WGD->allBuildingsV[bIdx]->base_height)
                                                  );

                    buildingTris.push_back(tri1);
                    buildingTris.push_back(tri2);
                }

            } //end of for all building polygon vertices

            // then create triangulated roof
            // requires walking through the base_height + height_eff
            // polygon plane and creating triangles...

            // triangle fan starting at vertice 0 of the polygon

            //Base Point (take the first polyvert edge and "fan" around
            Vector3<float> baseRoofPt(WGD->allBuildingsV[bIdx]->polygonVertices[0].x_poly,
                                      WGD->allBuildingsV[bIdx]->polygonVertices[0].y_poly,
                                      WGD->allBuildingsV[bIdx]->height_eff);

            for (auto pIdx=1u; pIdx < WGD->allBuildingsV[bIdx]->polygonVertices.size()-1; pIdx++){

                Triangle* triRoof = new Triangle(baseRoofPt,
                                                 Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[pIdx].x_poly,
                                                                WGD->allBuildingsV[bIdx]->polygonVertices[pIdx].y_poly,
                                                                WGD->allBuildingsV[bIdx]->height_eff),

                                                 Vector3<float>(WGD->allBuildingsV[bIdx]->polygonVertices[pIdx+1].x_poly,
                                                                WGD->allBuildingsV[bIdx]->polygonVertices[pIdx+1].y_poly,
                                                                WGD->allBuildingsV[bIdx]->height_eff));

                buildingTris.push_back(triRoof);

            } //end of roof for loop
        }
    }


    std::vector< Triangle* > groundTris(2);
    if (WID->simParams->DTE_heightField == nullptr) {

        // need to make sure we add the ground plane triangles.  There
        // // is no DEM in this case.
        groundTris[0] = new Triangle( Vector3<float>(0.0, 0.0, 0.0), Vector3<float>(nx*dx, 0.0f, 0.0f), Vector3<float>(nx*dx, ny*dy, 0.0f) );
        groundTris[1] = new Triangle( Vector3<float>(0.0, 0.0, 0.0), Vector3<float>(nx*dx, ny*dy, 0.0f), Vector3<float>(0.0f, ny*dy, 0.0f) );
    }

    // Assemble list of all triangles and create the mesh BVH
    std::cout << "Forming Length Scale triangle mesh...\n";
    std::vector<Triangle*> allTriangles;
    if (WID->simParams->DTE_heightField) {

        allTriangles.resize( WID->simParams->DTE_heightField->getTris().size() );

        //std::copy(WID->simParams->DTE_heightField->getTris().begin(), WID->simParams->DTE_heightField->getTris().end(), allTriangles.begin());

        for(int i = 0; i < WID->simParams->DTE_heightField->getTris().size(); i++){
            allTriangles[i] = WID->simParams->DTE_heightField->getTris()[i];
        }

    }
    else {
        allTriangles.insert(allTriangles.end(), groundTris.begin(), groundTris.end());
    }

    // Add building triangles to terrain and/or ground triangle
    for(int i = 0; i < buildingTris.size(); i++){
        allTriangles.push_back(buildingTris[i]);
    }

    Mesh *m_mixingLengthMesh = new Mesh(allTriangles);
    std::cout << "Triangle Meshing complete\n";

       
#ifdef HAS_OPTIX
    //TODO: Find a better way to get the list of Triangles
    // Will need to use ALL triangles vector rather than the DTE
    // mesh of triangles...
    //OptixRayTrace optixRayTracer(WID->simParams->DTE_mesh->getTris());
    
    std::cout<<"--------------------Before OptiX calls-------------------------"<<std::endl;
    OptixRayTrace optixRayTracer(m_mixingLengthMesh->getTris());
    optixRayTracer.calculateMixingLength( WID->localMixingParam->mlSamplesPerAirCell, nx, ny, nz, dx, dy, dz, WGD->icellflag, WGD->mixingLengths);
    
    std::cout<<"--------------------End of OptiX calls-------------------------"<<std::endl;
#else
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "NO OPTIX SUPPORT!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
#endif
    
    if(WID->localMixingParam->save2file){
        saveMixingLength(WID,WGD);
    }
    
    return;
}
