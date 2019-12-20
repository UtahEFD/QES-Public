#include "Ray.h"

Ray::Ray(float origin_x, float origin_y, float origin_z, Vector3<float> dirVec){
   this->origin_x = origin_x;
   this->origin_y = origin_y;
   this->origin_z = origin_z;
   this->dirVec = dirVec;
}

Ray::Ray(float origin_x, float origin_y, float origin_z){
   this->origin_x = origin_x;
   this->origin_y = origin_y;
   this->origin_z = origin_z;
}

float Ray::getOriginX(){return origin_x;}
float Ray::getOriginY(){return origin_y;}
float Ray::getOriginZ(){return origin_z;}
Vector3<float> Ray::getDirection() {return dirVec;}

void Ray::setDir(Vector3<float> dir){
   dirVec = dir;
}
