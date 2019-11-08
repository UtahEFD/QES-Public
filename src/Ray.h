/*
 *Basic definition of a ray 
 */
#include "Vector3.h"

class Ray{
  private:
      float origin_x, origin_y, origin_z;
      Vector3<float> dirVec;
  public:
      Ray(float origin_x, float origin_y, float origin_z, Vector3<float> dirVec){
         this.origin_x = origin_x;
         this.origin_y = origin_y;
         this.origin_z = origin_z;
         this.dirVec = dirVec;
      }

      Ray(float origin_x, float origin_y, float origin_z){
         this.origin_x = origin_x;
         this.origin_y = origin_y;
         this.origin_z = origin_z;
      }

      float getOriginX(){return origin_x;}
      float getOriginY(){return origin_y;}
      float getOriginZ(){return origin_z;}
      Vector3<float> getDirection() {return dirVec;}

      void setDir(Vector3<float> dir){dirVec = dir;}
};
