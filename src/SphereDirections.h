#pragma once

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
   int numDirVec;               //Number of directional vectors
   int vecCount;                //The number of vectors that have been currently generated 

   //range of sphere for random version
   float lowerThetaBound;
   float upperThetaBound;
   float lowerPhiBound;
   float upperPhiBound;
   
   Vector3<float> nextList[6];  //holds vectors of the 6 cardinal directions

  public:

   /*
    *Default constuctor for the 6 cardinal directions 
    */
   SphereDirections();

   /*
    *Constuctor for the random version
    */
   SphereDirections(int numDirVec, float lowerThetaBound, float upperThetaBound, float lowerPhiBound, float upperPhiBound);


   /*
    *@return numDirVec -the number of directional vectors generated
    */
   int getNumDirVec();

   /*
    *@return the next cardinal directional vector or NULL if the vecCount > numDirVec
    */
   Vector3<float> getNextDirCardinal();

   /*
    *Gets a randomly generated directional vector based on theta and
    *phi bounds
    *
    *@return the next randomly generated directional vector 
    */
   Vector3<float> getNextDir();

   /*
    *Gets the next unique direction 
    *@return the next non-repeated directional vector
    */
   Vector3<float> getNextDir2();
   
};

#endif

