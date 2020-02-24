#pragma once

/*
 * This class serves as a container for a BVH of triangles that 
 * represents a connected collection of Triangles
 */

#include "Triangle.h"
#include "BVH.h"
#include "SphereDirections.h"
#include "Ray.h"
#include "HitRecord.h"
//#include "OptixRayTrace.h"

#include <limits>
#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <iostream>

using std::vector;

class Mesh
{
public:
	BVH* tris;


        //temp var for Optix
        //OptixRayTrace *optixRayTracer;
        vector<Triangle*> optixTris;
        
	/*
	 * Creates a BVH out of a list of Triangles
	 *
	 * @param tris -list of triangles.
	 */
	Mesh(vector<Triangle*> tris)
	{
		this->tris = BVH::createBVH(tris);

                //temp var for Optix
                // this->optixRayTracer = new OptixRayTrace(tris);
                       optixTris = tris;
        }

	/*
	 * Gets the height from a location on the xy plane
	 * to a triangle in the BVH
	 *
	 * @param x -x position
	 * @param y -y position
	 * @return distance to the triangle directly above the point.
	 */
	float getHeight(float x, float y);


        /*
         *Calculates the mixing length for all fluid objects
         *
         *@param dimX -domain info in the x plane 
         *@param dimY -domain info in the y plane
         *@param dimZ -domain info in the z plane 
         *@param dx -grid info in the x plane
         *@param dy -grid info in the y plane
         *@param dz -grid info in the z plane
         *@param icellflag -cell type
         *@param mixingLengths -array of mixinglengths for all cells that will be updated 
         */
        void calculateMixingLength(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const vector<int> &icellflag, vector<double> &mixingLengths);
        
        void tempOPTIXMethod(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const vector<int> &icellflag, vector<double> &mixingLengths){
           //         OptixRayTrace optixRayTracer(optixTris); 
           //optixRayTracer.calculateMixingLength(10, dimX, dimY, dimZ, dx, dy, dz, icellflag, mixingLengths);
        }


        //temporary getter for the list of Triangles
        vector<Triangle *> getMeshTriList(){
           return optixTris;
        }
};
