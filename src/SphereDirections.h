#ifndef SDIR_H
#define SDIR_H

/*
 *Class used to generate direction vectors for a sphere to use in ray
 *tracing 
 */

#include Vector3.h

class SphereDirections{
  private:
   int numDirVec = 6; //Number of directional vectors
   int vecCount = 0;
   float disBtwVectors;
   Vector3 bound_x;
   Vector3 bound_y;
   Vecotr3 bound_z; 
   Vector3 next; //next directional vector
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
    *Calculates the next directional vector
    *updates local parameters to reflect the next direction it will calculate 
    *Returns the next directional vector if vecCount <= numDirVec
    *     returns NULL otherwise
    */
   Vector3<float> calcNextVector();
   
   
};

#endif
