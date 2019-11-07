/*
 *Basic definition of a ray 
 */
#include "Vector3.h"

class Ray{
  private:
   Vector3 originVec, dirVec;
  public:
   Ray(Vector3<float> originVec, Vector3<float> dirVec);

   Vector3<float> getOrigin() {return originVec;}
   Vector3<float> getDirection() {return dirVec;}
};
