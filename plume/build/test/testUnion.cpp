 
#include <cstdio>
// #include <cutil_math.h>
#include "vector_types.h" 

enum SourceType{LINE, SPHERE};

struct sphere
{
  float3 ori;
  int rad;
} ;

struct line
{
  float3 start;
  float3 end;
};

union sourceinfo
{
  sphere sp;
  line ln;
};
 

struct source
{
  SourceType st;
//   int sourceOri;
  sourceinfo si;
};


int main()
{
  source sc;
  sc.st = LINE;
  float3 d;
  d.x=1;d.y=2;d.z=3;
  sc.si.ln.start.x =1 ;
  sc.si.ln.start.y =2 ;
  sc.si.ln.start.z =3 ;
  
  sc.si.ln.end.x =1 ;
  sc.si.ln.end.y =2 ;
  sc.si.ln.end.z =3 ;//make_float3(0.f,0.f,0.f);
   
  printf("%d,  %d\n", sizeof(sphere), sizeof(line));
  
}