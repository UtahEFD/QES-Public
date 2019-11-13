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
               BVH* nextBVH;

               //for all possible directions, determine the distance
               for(int m = 0; m < sd.getNumDirVec(); m++){

                  ray.setDir(sd.getNextDir());
                  nextBVH = tris;
                  hit = NULL;

                  //stop if nextBVH is a leaf node or if it didn't intersect anything
                  while(!(nextBVH->getIsLeaf()) && nextBVH != 0){
                     HitRecord *checkLeft = (nextBVH->getLeftBox())->rayBoxIntersect(ray);
                     HitRecord *checkRight = (nextBVH->getRightBox())->rayBoxIntersect(ray);

                     if(checkLeft != 0 && checkRight == 0){
                        nextBVH = nextBVH->getLeftBox();
                     }else if(checkLeft == 0 && checkRight != 0L){
                        nextBVH = nextBVH->getRightBox();
                     }else if (checkLeft != 0 && checkRight != 0){
                        if(checkLeft->getHitDist() < checkRight->getHitDist()){
                           nextBVH = nextBVH->getLeftBox();
                        }else{
                           nextBVH = nextBVH->getRightBox();
                        }
                     }else{
                        std::cout<<"Ray is outside of bounds. Did not hit any of the bounding boxes"<<std::endl;
                        nextBVH = 0;
                     }
                  }

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
               }
               std::cout<<"Mixing length for this cell is"<<mixLength<<std::endl;
               //add to list of vectors
               mixingLengthList.push_back(mixLength);
            }

         }
      }
   }
   return mixingLengthList;
}
