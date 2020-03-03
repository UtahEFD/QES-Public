#pragma once

#ifndef SDIR_H
#define SDIR_H

/*
 *Class used to generate direction vectors for a sphere to use in ray
 *tracing 
 */

#include "Vector3.h"
#include <cmath>
#include <cfloat>
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
   Vector3<float> prevDir;
  public:

   /*
    *Default constuctor for the 6 cardinal directions 
    */
   SphereDirections();

   /*
    *Constuctor for the random version
    */
   SphereDirections(int numDirVec, float lowerThetaBound, float upperThetaBound, float lowerPhiBound, float upperPhiBound);


   /*Constructor for the Mitchell's Best Candidate Algorithm test 
    */
   SphereDirections(int numDirVec);
   

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

   /*Mitchel's Best Algorithm 
    *Gets the next unique direction 
    *@return the next non-repeated directional vector
    */
   Vector3<float> getNextDir2();
   Vector3<float> genRandSpherePt();
   Vector3<float> polarToCartesian(float theta, float phi);
   Vector3<float> furthestPoint(Vector3<float> *samplePtList, Vector3<float> existingPt, const int listSize);
};

#endif

