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



HitRecord* BVH::rayTriangleIntersect(Ray* ray){
   HitRecord hitrec = NULL;
   if(!isleaf){
      std::cout<<"Not an intersectable triangle."<<endl;
   }else{
      float beta, gamma, t, M, a,b,c,d,e,f,g,h,i,j,k,l;

      a = (tri.a)[0] - (tri.a)[1];       d = (tri.a)[0] - (tri.a)[2];   g = ray.getDirection()[0];
      b = (tri.b)[0] - (tri.b)[1];       e = (tri.b)[0] - (tri.b)[2];   h = ray.getDirection()[1];
      c = (tri.c)[0] - (tri.c)[1];       f = (tri.c)[0] - (tri.c)[2];   i = ray.getDirection()[2];

      j = (tri.a)[0] - ray.getOriginX();
      k = (tri.b)[0] - ray.getOriginY();
      l = (tri.c)[0] - ray.getOriginZ();

      float EIHF = (e*i) - (h*f);
      float GFDI = (g*f) - (d*i);
      float DHEG = (d*h) - (e*g);
      float JCAL = (j*c) - (a*l);
      float BLKC = (b*l) - (k*c);
      float AKJB = (a*k) - (j*b);

      M = (a*EIHF) + (b*GFDI) + (c*DHEG);

      beta = ((j*EIHF) + (k*GFDI) + (l*DHEG))/M;
      gamma = ((i*AKJB) + (h*JCAL) + (g*BLKC))/M;
      t = ((f*AKJB) + (e*JCAL) + d(BLKC))/M;

      if(t < c || t > f || gamma < 0 || gamma > 1 || beta < 0 || beta > (1- gamma)){
         std::cout<<"No intersection found."<<endl;
      }else{
         hitrec = new HitRecord(tri, t);
      }

   }

   return &hitrec;
}


HitRecord* BVH::rayBoxIntersect(Ray* ray){
   HitRecord hitrec = NULL;

   if(isLeaf){
      cout<<"This is a leaf node"<<endl;
   }else{
      float rOriginX = ray.getOriginX();
      float rOriginY = ray.getOriginY();
      float rOriginZ = ray.getOriginZ();

      Vector3<float> toTop = std::abs(rOriginZ-zmax)*ray.getDirection();
      Vector3<float> ptOnTop = new Vector3(rayOriginX +toTop, rayOriginY + toTop, rayOriginZ + toTop);

      Vector3<float> toBottom = std::abs(rOriginZ -zmin)*ray.getDirection();
      Vector3<float> ptOnBottom = new Vector3(rOriginX + toBottom, rOriginY + toBottom, rOriginZ + toBottom);

      Vector3<float> toLeft = std::abs(rOriginY - ymin) * ray.getDirection();
      Vector3<float> ptOnLeft = new Vector3(rOriginX + toLeft, rOriginY + toLeft, rOriginZ + toLeft);

      Vector3<float> toRight = std::abs(rOriginY - ymax)* ray.getDirection();
      Vector3<float> ptOnRight = new Vector3(rOriginX + toRight, rOriginY + toRight, rOriginZ + toRight);

      Vector3<float> toFront = std::abs(rOriginX - xmax)*r.getDirection();
      Vector3<float> ptOnFront = new Vector3(rOriginX + toFront, rOriginY + toFront, rOriginZ + toFront);

      Vector3<float> toBack = std::abs(rOriginX - xmin)*r.getDirection();
      Vector3<float> ptOnFront = new Vector3(rOriginX + toBack, rOriginY + toBack, rOriginZ +toBack);


      if(ptOnTop[2] == zmax && ptOnTop[0] >= xmin && ptOnTop[0] <= xmax && ptOnTop[1] >= ymin && ptOnTop[1] <= ymax){

         hitrec = new HitRecord(this, std::sqrt((ptToTop[0]*ptToTop[0])+(ptToTop[1]*ptToTop[1])+(ptToTop[2]*ptToTop[2])));

      }else if(ptOnLeft[1] == ymin && ptOnLeft[0] >= xmin && ptOnLeft[0] <= xmax && ptOnLeft[2] >= zmin && ptOnLeft[2] <= zmax){

         hitrec = new HitRecord(this, std::sqrt((ptToLeft[0]*ptToLeft[0])+(ptToLeft[1]*ptToLeft[1])+(ptToLeft[2]*ptToLeft[2])));

      }else if(ptOnFront[0] == xmax && ptToFront[1] >= ymin && ptToFront[1] <= ymax && ptToFront[2] >= zmin && ptToFront[2] <= zmax){

         hitrec = new HitRecord(this, std::sqrt((ptToFront[0]*ptToFront[0])+(ptToFront[1]*ptToFront[1])+(ptToFront[2]*ptToFront[2])));

      }else if(ptOnBottom[2] == zmin && ptOnBottom[0] >= xmin && ptOnBottom[0] <= xmax && ptOnBottom[1] >= ymin && ptOnBottom[1] <=ymax){

         hitrec = new HitRecord(this, std::sqrt((ptToBottom[0]*ptToBottom[0])+(ptToBottom[1]*ptToBottom[1])+(ptToBottom[2]*ptToBottom[2])));

      }else if(ptToRight[1] == ymax && pToRight[0] >= xmin && ptToRight <= xmax && ptToRight[2] >= zmin && ptToRight <= zmax){

         hitrec = new HitRecord(this, std::sqrt((ptToRight[0]*ptToRight[0])+(ptToRight[1]*ptToRight[1])+(ptToRight[2]*ptToRight[2])));

      }else if(ptToBack[0] == xmin && ptToBack[1] >= ymin && ptToBack[1] <= ymax && ptToBack[2] >= zmin && ptToBack[2] <= zmax){

         hitrec = new HitRecord(this, std::sqrt((ptToBack[0]*ptToBack[0])+(ptToBack[1]*ptToBack[1])+(ptToBack[2]*ptToBack[2])));

      }else{
         std::cout<<"Does not hit this BVH bounding box"<<endl;
      }

      return &hitrec;

   }
