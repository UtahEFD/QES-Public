#include "BVH.h"

void BVH::mergeSort(std::vector<BVH*>& list, const int type)
{
   if (list.size() <= 1)
      return;

   std::vector<BVH *> l, r;
   int midPoint = list.size() / 2;
   for (int i = 0; i < midPoint; i++)
      l.push_back(list[i]);
   for (int i = midPoint; i < list.size(); i++)
      r.push_back(list[i]);

   mergeSort(l, type);
   mergeSort(r, type);

   int lSize = l.size(), rSize = r.size();
   int j = 0, k = 0;
   for (auto i = 0u; i < list.size(); i++)
   {
      if ( j == lSize)
         list[i] = r[k++];
      else if ( k == rSize)
         list[i] = l[j++];
      else
      {
         float lMid, rMid;
         if (type == 0)
         {
            lMid = (l[j]->xmin + l[j]->xmax) / 2.0f;
            rMid = (r[k]->xmin + r[k]->xmax) / 2.0f;
         }
         else
         {
            lMid = (l[j]->ymin + l[j]->ymax) / 2.0f;
            rMid = (r[k]->ymin + r[k]->ymax) / 2.0f;
         }
         list[i] = (lMid < rMid ? l[j++] : r[k++]);
      }
   }

}

BVH::BVH(BVH* l, BVH* r)
{
   leftBox = l;
   rightBox = r;
   if (r != 0)
   {
      xmin = GETMIN(l->xmin, r->xmin);
      xmax = GETMAX(l->xmax, r->xmax);
      ymin = GETMIN(l->ymin, r->ymin);
      ymax = GETMAX(l->ymax, r->ymax);
      zmin = GETMIN(l->zmin, r->zmin);
      zmax = GETMAX(l->zmax, r->zmax);
   }
   else
   {
      xmin = l->xmin;
      xmax = l->xmax;
      ymin = l->ymin;
      ymax = l->ymax;
      zmin = l->zmin;
      zmax = l->zmax;
   }

   isLeaf = false;
   tri = 0;
}

BVH::BVH(Triangle* t)
{
   tri = t;
   isLeaf = true;

   t->getBoundaries(xmin, xmax, ymin, ymax, zmin, zmax);

   leftBox = rightBox = 0;
}

BVH::BVH(std::vector<BVH *> m, int height)
{
   isLeaf = false;
   if (m.size() == 1)
   {
      leftBox = m[0];
      rightBox = 0;

      xmin = leftBox->xmin;
      xmax = leftBox->xmax;
      ymin = leftBox->ymin;
      ymax = leftBox->ymax;
      zmin = leftBox->zmin;
      zmax = leftBox->zmax;

      tri = 0;
      return;
   }
   else if (m.size() == 2)
   {
      leftBox = m[0];
      rightBox = m[1];

      xmin = GETMIN(leftBox->xmin, rightBox->xmin);
      xmax = GETMAX(leftBox->xmax, rightBox->xmax);
      ymin = GETMIN(leftBox->ymin, rightBox->ymin);
      ymax = GETMAX(leftBox->ymax, rightBox->ymax);
      zmin = GETMIN(leftBox->zmin, rightBox->zmin);
      zmax = GETMAX(leftBox->zmax, rightBox->zmax);

      tri = 0;
      return;
   }

   mergeSort(m, height % 2);


   std::vector<BVH *> l, r;
   int midPoint = m.size() / 2;
   for (int i = 0; i < midPoint; i++)
      l.push_back(m[i]);
   for (int i = midPoint; i < m.size(); i++)
      r.push_back(m[i]);

   leftBox = new BVH(l, height + 1);
   rightBox = new BVH(r, height + 1);

   xmin = GETMIN(leftBox->xmin, rightBox->xmin);
   xmax = GETMAX(leftBox->xmax, rightBox->xmax);
   ymin = GETMIN(leftBox->ymin, rightBox->ymin);
   ymax = GETMAX(leftBox->ymax, rightBox->ymax);
   zmin = GETMIN(leftBox->zmin, rightBox->zmin);
   zmax = GETMAX(leftBox->zmax, rightBox->zmax);

   tri = 0;
}


BVH* BVH::createBVH(const std::vector<Triangle*> tris)
{
   std::vector<BVH *> boxes;

   for (int i = 0; i < tris.size(); i++)
   {
      BVH* b = new BVH(tris[i]);
      boxes.push_back(b);
   }

   BVH* root = new BVH(boxes, 0);
   return root;
}

float BVH::heightToTri(float x, float y)
{
   if (isLeaf)
   {
      return tri->getHeightTo(x, y);
   }
   else
   {
      float toL = -1.0f, toR = -1.0f;

      if (leftBox && leftBox->xmin <= x && leftBox->xmax >= x &&
          leftBox->ymin <= y && leftBox->ymax >= y)
         toL = leftBox->heightToTri(x, y);

      if (rightBox && rightBox->xmin <= x && rightBox->xmax >= x &&
          rightBox->ymin <= y && rightBox->ymax >= y)
         toR = rightBox->heightToTri(x, y);

      return toL > toR ? toL : toR;
   }
}


bool BVH::rayBoxIntersect(Ray ray){
   float originX = ray.getOriginX();
   float originY = ray.getOriginY();
   float originZ = ray.getOriginZ();
   Vector3<float> dir = ray.getDirection();

   float tMinX, tMaxX, tMinY, tMaxY, tMinZ, tMaxZ;

   //calc tMinX and tMaxX
   float aX = 1/dir[0];
   if(aX >= 0){
      tMinX = aX*(xmin - originX);
      tMaxX= aX*(xmax - originX);
   }else{
      tMinX = aX*(xmax - originX);
      tMaxX = aX*(xmin - originX);
   }

   //calc tMinY and tMaxY
   float aY = 1/dir[1];
   if(aY >= 0){
      tMinY = aY*(ymin - originY);
      tMaxY = aY*(ymax - originY);
   }else{
      tMinY = aY*(ymax - originY);
      tMaxY = aY*(ymin - originY);
   }

   //calc tMinZ and tMaxZ
   float aZ = 1/dir[2];
   if(aZ >= 0){
      tMinZ = aZ*(zmin - originZ);
      tMaxZ = aZ*(zmax - originZ);
   }else{
      tMinZ = aZ*(zmax - originZ);
      tMaxZ = aZ*(zmin - originZ);
   }


   if(tMinX > tMaxY || tMinY > tMaxX ||
      tMinY > tMaxZ || tMinZ > tMaxY ||
      tMinZ > tMaxX || tMinX > tMaxZ){
      return false;
   }else{
      return true;
   }
}


bool BVH::rayHit(Ray ray, const float t0, float& t1, HitRecord& rec){
   if(!rayBoxIntersect(ray)){
      //std::cout<<"didn't intersect parent"<<std::endl;
      return false;
   }else{
      if(isLeaf){
         //init rec
         if(!tri){
            std::cout<<"Must not be a tri."<<std::endl;
         }
         bool triHit = tri->rayTriangleIntersect(ray, rec);
         //std::cout<<"triHit = "<<triHit<<std::endl;
         return triHit; //if false, then it didn't intersect the primitative
      }else{  //not a leaf node
         bool leftHit = false, rightHit = false;
         HitRecord lrec, rrec;

         //prob don't need to check left due to BVH construction
         //left should never be null for anything except a leaf node
         if(leftBox){
            leftHit = leftBox->rayHit(ray, t0, t1, lrec);
         }

         if(rightBox){
            rightHit = rightBox->rayHit(ray, t0, t1, rrec);
         }

         if(leftHit && rightHit){
            if(lrec.hitDist < rrec.hitDist){
               rec = lrec;
            }else{
               rec = rrec;
            }
         }else if(leftHit){
            rec = lrec;
         }else if(rightHit){
            rec = rrec;
         }else{
            return false;
         }
      }
   }
}



bool BVH::getIsLeaf(){return isLeaf;}
BVH* BVH::getLeftBox(){return leftBox;}
BVH* BVH::getRightBox(){return rightBox;}
