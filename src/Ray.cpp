#include "Ray.h"

Ray::Ray(float origin_x, float origin_y, float origin_z, Vector3<float> dirVec){
   this->origin_x = origin_x;
   this->origin_y = origin_y;
   this->origin_z = origin_z;
   this->dirVec = dirVec;

   inverse_dir = Vector3<float>(1/dirVec[0], 1/dirVec[1], 1/dirVec[2]);
   signs.push_back(inverse_dir[0] < 0);
   signs.push_back(inverse_dir[1] < 0);
   signs.push_back(inverse_dir[2] < 0);
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

Vector3<float> Ray::getInverseDir(){return inverse_dir;}
std::vector<int> Ray::getSigns(){return signs;};

void Ray::setDir(Vector3<float> dir){
   dirVec = dir;
   inverse_dir = Vector3<float>(1/dirVec[0], 1/dirVec[1], 1/dirVec[2]);
   signs.push_back(inverse_dir[0] < 0);
   signs.push_back(inverse_dir[1] < 0);
   signs.push_back(inverse_dir[2] < 0);
}
