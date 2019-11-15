#include "Mesh.h"

float Mesh::getHeight(float x, float y)
{
   return tris->heightToTri(x,y);
}

vector<float> Mesh::calculateMixingLength(int dimX, int dimY,int dimZ, float dx, float dy, float dz, const vector<int> &icellflag){

   vector<float> mixingLengthList(dimX*dimY*dimZ);

   for(int k = 0; k< dimZ - 1; k++) {
      for(int j = 0; j < dimY - 1; j++){
         for(int i = 0; i < dimX -1; i++){

            //calculate icell index
            int icell_idx = i + j*(dimX-1) + k*(dimY-1) * (dimX-1);

            if(icellflag[icell_idx] == 1){

               SphereDirections sd;
               float mixLength = std::numeric_limits<float>::infinity();

               //ray's origin = cell's center
               Ray ray((i+0.5)*dx, (j+0.5)*dy, (k+0.5)*dz);


               HitRecord* hit;

               //BVH* nextBVH;

               //for all possible directions, determine the distance
               for(int m = 0; m < sd.getNumDirVec(); m++){
                  std::cout<<"\n\nDirection interation #: "<<m<<std::endl;
                  ray.setDir(sd.getNextDir());
                  //nextBVH = tris;
                  //std::cout<<"nextBVH = "<<nextBVH<<std::endl;
                  hit = NULL;

                  bool isHit = tris->rayHit(ray, hit);

                  if(isHit){
                     std::cout<<"Hit found."<<std::endl;

                     //compare the mixLengths
                     if(hit->getHitDist() < mixLength){
                        mixLength = hit->getHitDist();
                     }
                  }else{
                     std::cout<<"Hit not found"<<std::endl;
                  }

                  /*HitRecord *checkLeft;
                    HitRecord *checkRight;

                    //stop if nextBVH is a leaf node or if it didn't intersect anything
                    while(!(nextBVH->getIsLeaf()) && nextBVH != NULL){

                    if(nextBVH->getLeftBox() == NULL || nextBVH->getRightBox() == NULL){
                    std::cout<<"One/both of the boxes were NULL"<<std::endl;
                    }else{
                    checkLeft = (nextBVH->getLeftBox())->rayBoxIntersect(ray);
                    checkRight = (nextBVH->getRightBox())->rayBoxIntersect(ray);

                    std::cout<<"Check left = "<<checkLeft->getHitNode()<<"\tisHit ="<<checkLeft->getIsHit()<<std::endl;
                    std::cout<<"Check right = "<<checkRight->getHitNode()<<"\tisHit "<<checkRight->getIsHit()<<std::endl;
                    if(checkLeft->getIsHit() && !(checkRight->getIsHit())){
                    nextBVH = nextBVH->getLeftBox();
                    std::cout<<"Enters left box condition.\t dist = "<<checkLeft->getHitDist()<<std::endl;

                    }else if(!(checkLeft->getIsHit()) && checkRight->getIsHit()){
                    nextBVH = nextBVH->getRightBox();
                    //std::cout<<"Enters right box condition.\tdist = "<< checkRight->getHitDist() <<std::endl;

                    }else if (checkLeft->getIsHit() && checkRight->getIsHit()){
                    std::cout<<"Distance of left = "<<checkLeft->getHitDist()<<"\tright dist = "<<checkRight->getHitDist()<<std::endl;
                    if(checkLeft->getHitDist() < checkRight->getHitDist()){
                    std::cout<<"Chose the left box as the one with the shortest distance"<<std::endl;
                    nextBVH = nextBVH->getLeftBox();
                    }else{
                    std::cout<<"Chose the right box as the one witht he shortest distance."<<std::endl;
                    nextBVH = nextBVH->getRightBox();
                    }
                    }else{
                    std::cout<<"Ray is outside of bounds. Did not hit any of the bounding boxes"<<std::endl;
                    std::cout<<"isLeaf after bound fail = "<<nextBVH->getIsLeaf()<<std::endl;
                    nextBVH = 0;
                    }

                    std::cout<<"isLeaf for the new nextBVH = "<<nextBVH->getIsLeaf()<<std::endl;
                    std::cout<<"nextBVH after assignment = "<<nextBVH<<std::endl;
                    }
                    }

                    std::cout<<"!!!Finished finding leaf node. isLeaf = "<<nextBVH->getIsLeaf()<<std::endl;

                    //At this point BVH should be a leaf or NULL
                    //if nextBVH is a leaf, it should check if it's the smallest
                    //dist currently
                    if(nextBVH != 0){
                    hit = nextBVH->rayTriangleIntersect(ray);
                    if(nextBVH != 0 & hit != 0){
                    if(hit->getHitDist() < mixLength){
                    mixLength = hit->getHitDist();
                    }
                    }
                    }
                  */
               }

               std::cout<<"Mixing length for this cell is"<<mixLength<<std::endl;
               //add to list of vectors
               mixingLengthList.push_back(mixLength);
               std::cout<<"\n\n"<<std::endl;
            }

         }
      }
   }
   return mixingLengthList;
}
