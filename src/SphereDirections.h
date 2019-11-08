#ifndef SDIR_H
#define SDIR_H

/*
 *Class used to generate direction vectors for a sphere to use in ray
 *tracing 
 */

#include "Vector3.h"

class SphereDirections{
  private:
   int numDirVec; //Number of directional vectors
   int vecCount;
   float disBtwVectors;
   Vector3<float> bound_x;
   Vector3<float> bound_y;
   Vector3<float> bound_z;
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
   SphereDirections(int numDir, Vector3<float> bound_vec_x,
                    Vector3<float> bound_vec_y, Vector3<float> bound_vec_z);


   /*
    *@return numDirVec The number of directional vectors generated
    */
   int getNumDirVec(){return numDirVec;}

   /*
    *Calculates the next directional vector
    *updates local parameters to reflect the next direction it will calculate 
    *Returns the next directional vector if vecCount <= numDirVec
    *     returns NULL otherwise
    */
   Vector3<float> getNextDir();
   
   
};

#endif

