/*
 *Basic definition of a ray 
 */
#include "Vector3.h"

class Ray{
  private:
      float origin_x, origin_y, origin_z;
      Vector3<float> dirVec;
  public:
      Ray(float origin_x, float origin_y, float origin_z, Vector3<float> dicVec);

      Ray(float origin_x, float origin_y, float origin_z);

      float getOriginX();
      float getOriginY();
      float getOriginZ();
      Vector3<float> getDirection();

      void setDir(Vector3<float> dir);
};
