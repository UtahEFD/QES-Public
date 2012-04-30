/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Provide a means to move about a 3D space.
*/

#ifndef INC_VIEWER_H
#define INC_VIEWER_H

#include "normal.h"

/**
*	Needs some work.
*/
class viewer 
{
  public:
                
    point eye;
    point foc;
    normal up;
        
    //Constructors
        
    viewer(void);
    viewer(viewer const& v);
    viewer(point const& loc);
    viewer(point const& loc, point const& foc);
    viewer(point const& loc, point const& foc, normal const& up);
        
    //Methods
    point getEye() const;
    void setEye(point const& loc);
    void shiftEye(point const& disp);

    point getFocus() const;
    void setFocus(point const& foc);
    void shiftFocus(point const& disp);

    normal getGaze() const;
    void setGaze(normal const& g);

    normal getRight() const;
    normal getUp() const;
    void setUp(normal const& n);

        		
    void lookRight(float const& deg);
    void lookUp(float const& deg);
    void rollRight(float const& deg);

    void lookLeft(float const& deg);
    void lookDown(float const& deg);
    void rollLeft(float const& deg);

     
    void moveForward(float const& disp);
    void strafeRight(float const& disp);
    void strafeUp(float const& disp);

    void moveBackward(float const& disp);
    void strafeLeft(float const& disp);
    void strafeDown(float const& disp);

    void operator =(viewer const& v);
};

void gluLookAt(point const& e, point const& f, normal const& u);

#endif

