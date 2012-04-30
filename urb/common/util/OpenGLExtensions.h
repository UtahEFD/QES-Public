/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Make OpenGL calls for the point and normal class easier.
*/
#ifndef OPENGL_EXTENSIONS
#define OPENGL_EXTENSIONS 1

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else 
#include <GL/glu.h>
#endif

#include "normal.h"

inline void glVertex3f(point const& pnt) 
{
  glVertex3f(pnt.x, pnt.y, pnt.z);
}

inline void glTranslatef(point const& pnt)
{
  glTranslatef(pnt.x, pnt.y, pnt.z);
}

inline void glNormal3f(normal const& nrml) 
{
  glNormal3f(nrml.getX(), nrml.getY(), nrml.getZ());
}


#endif
