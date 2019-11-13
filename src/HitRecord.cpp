#include "HitRecord.h"

HitRecord::HitRecord(void* hitNode, hitDist){
   this.hitNode = hitNode;
   this.hitDist = hitDist;
}

//BVH* HitRecord::getHitNode(){return hitnode;}
void* HitRecord::getHitNode(){return hitnode;}
float HitRecord::getHitDist(){return hitDist;}
