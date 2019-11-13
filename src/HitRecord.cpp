#include "HitRecord.h"

HitRecord::HitRecord(void* hitNode, float hitDist){
   this->hitNode = hitNode;
   this->hitDist = hitDist;
}

void* HitRecord::getHitNode(){return hitNode;}
float HitRecord::getHitDist(){return hitDist;}
