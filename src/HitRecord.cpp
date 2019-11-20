#include "HitRecord.h"

HitRecord::HitRecord(){
   //isHit = false;
   //hitNode = 0;
   //hitDist = -1;
}

HitRecord::HitRecord(void* hitNode, bool isHit){
   this->hitNode = hitNode;
   this->isHit = isHit;
   hitDist = -1*(std::numeric_limits<float>::infinity());
}

HitRecord::HitRecord(void* hitNode, bool isHit, float hitDist){
   this->hitNode = hitNode;
   this->isHit = isHit;
   this->hitDist = hitDist;
}

void* HitRecord::getHitNode(){return hitNode;}
float HitRecord::getHitDist(){return hitDist;}
bool HitRecord::getIsHit(){return isHit;}
