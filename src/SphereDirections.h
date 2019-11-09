#ifndef SDIR_H
#define SDIR_H

/*
 *Class used to generate direction vectors for a sphere to use in ray
 *tracing 
 */

#include "Vector3.h"
#include <cmath>
#include <random>

class SphereDirections{
  private:
   int numDirVec; //Number of directional vectors
   int vecCount;
   float disBtwVectors;
   float lowerThetaBound;
   float upperThetaBound;
   float lowerPhiBound;
   float upperPhiBound;
   Vector3<float> nextList[6];  //temp for 6 cardinal directions 

  //Vector3 next; //next directional vector
  public:

   /*Constructor
    *will generate directional vectors based on default numDir
    */
   SphereDirections();

   /*Constructor
    *will generate specified num of directional vectors bounded by a certain region
    */
   SphereDirections(int numDir, float lowerThetaBound, float upperThetaBound, float lowerPhiBound, float upperPhiBound);


   /*
    *@return numDirVec The number of directional vectors generated
    */
   int getNumDirVec(){return numDirVec;}

   /*
    *Returns the next cardinal directional vector
    *Returns NULL if the vecCount > numDirVec
    */
   Vector3<float> getNextDirCardinal();

   /*
    *Gets a randomly generated directional vector based on theta and
    *phi bounds
    */
   Vector3<float> getNextDir();
   
};

#endif

