#pragma once

/*
 *Class used to generate direction vectors for a sphere to use in ray
 *tracing 
 */

#include <cmath>
#include <cfloat>
#include <random>
#include "Vec3D.h"

class SphereDirections
{
private:
    int vecCount;
    int numDirs;
    
    //range of sphere for random version
    float lowerThetaBound;
    float upperThetaBound;
    float lowerPhiBound;
    float upperPhiBound;

    // std::vector< Vec3D > nextList; // [6];  //holds vectors of the 6

    Vec3D *nextList; 

  public:

   /*
    *Default constuctor for the 6 cardinal directions 
    */
    SphereDirections()
        : vecCount(0)
    {
        nextList = new Vec3D[ 6 ];
        
        //default cardinal directions for now
        nextList[0] = Vec3D(1,0,0);   //front
        nextList[1] = Vec3D(-1,0,0);  //back
        nextList[2] = Vec3D(0,1,0);   //left
        nextList[3] = Vec3D(0,-1,0);  //right
        nextList[4] = Vec3D(0,0,1);   //top
        nextList[5] = Vec3D(0,0,-1);  //bottom

        numDirs = 6;
    }

   /*
    *Constuctor for the random version
    */
   SphereDirections(int numDV, float lowerTheta, float upperTheta, float lowerPhi, float upperPhi)
       : vecCount(0), lowerThetaBound( lowerTheta ), upperThetaBound( upperTheta ), lowerPhiBound(lowerPhi), upperPhiBound(upperPhi)

    {
//        std::random_device rd;  // the rd device reads from a file,
//        apparently and thus, calls strlen, might need some other way
//        to seed.
        
//        std::mt19937 e2(rd());
        std::mt19937 e2(303);
        std::uniform_real_distribution<float> theta(lowerThetaBound, upperThetaBound);
        std::uniform_real_distribution<float> phi(lowerPhiBound, upperPhiBound);

        numDirs = numDV + 5;
        nextList = new Vec3D[ numDirs ];

        // for (int i=0; i<numDV; i++) {
        int i = 0;
        while (i < numDV) {

            float theta2 = std::asin(theta(e2));

            float dx = std::cos(theta2)*std::cos(phi(e2));
            float dy = std::sin(phi(e2));
            float dz = std::cos(theta2)*std::sin(phi(e2));
            
            float magnitude = std::sqrt(dx*dx + dy*dy + dz*dz);

            // only send rays mostly down but a little up... can use
            // dot product between (0, 0, 1) and vector
            Vec3D dirVec( dx/magnitude,dy/magnitude,dz/magnitude );

            float dotProd = dirVec[0]*0.0f + dirVec[1]*0.0f + dirVec[2]*1.0f;
                
            //if (dotProd < 0.20) {
                nextList[i] = Vec3D(dx/magnitude,dy/magnitude,dz/magnitude);
                i++;
                // }
        }
        
        // Then make sure the cardinal directions that may matter are
        // added -- up is unlikely at this point
        nextList[numDV] = Vec3D(1,0,0);   //front
        nextList[numDV + 1] = Vec3D(-1,0,0);  //back
        nextList[numDV + 2] = Vec3D(0,1,0);   //left
        nextList[numDV + 3] = Vec3D(0,-1,0);  //right
        nextList[numDV + 4] = Vec3D(0,0,-1);  //bottom

//        std::cout << "Generated " << nextList.size() << " sphere directions." << std::endl;
//        std::cout << "sd = [" << std::endl;
//        for (int i=0; i<nextList.size(); i++) {
//            std::cout << "\t" << nextList[i][0] << " " << nextList[i][1] << " " << nextList[i][2] << ";" << std::endl;
//        }
//        std::cout << "];" << std::endl;

        
    }
    

    ~SphereDirections() 
    {
        delete [] nextList;
    }
    

   /*Constructor for the Mitchell's Best Candidate Algorithm test 
    */
   SphereDirections(int numDirVec);
   

   /*
    *@return numDirVec -the number of directional vectors generated
    */
    int getNumDirVec() { return numDirs; } 

   /*
    *@return the next cardinal directional vector or NULL if the vecCount > numDirVec
    */
//   Vector3<float> getNextDirCardinal();

   /*
    *Gets a randomly generated directional vector based on theta and
    *phi bounds
    *
    *@return the next randomly generated directional vector 
    */
//   Vector3<float> getNextDir();
    Vec3D getNextDir() 
    {
        Vec3D retVal = nextList[ vecCount ];
        
        vecCount++;
        if (vecCount > numDirs)
            vecCount = numDirs-1;
        
        return retVal;
    }
    

   /*Mitchel's Best Algorithm 
    *Gets the next unique direction 
    *@return the next non-repeated directional vector
    */
//   Vector3<float> getNextDir2();
   
};

