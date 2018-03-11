/* File: VolumeCell.cpp
 * Author: Matthew Overby
 */

#include "VolumeCell.h"

using namespace SLUI;

VolumeCell::VolumeCell(){

	volumeType = -1;
	renderable = false;
}

VolumeCell::VolumeCell( struct TreeCell tc ){

	volumeType = VolumeType::Tree;
	renderable = true;
	treeData = tc;
}


VolumeCell::~VolumeCell(){}

//	This is a super-ishy way to draw trees.  It's just a temporary
//	function to draw these basic shapes until needs are more defined
//
void VolumeCell::draw(){

	if( !renderable ) return;

	switch( volumeType ){

		case VolumeType::Tree:{

			vector3D brown = vector3D( 0.58f, 0.2f, 0.1f );
			vector3D green = vector3D( 0.f, 0.4f, 0.f );

			if( treeData.shape == TreeType::Shape::Rocket ){
				vector3D green = vector3D( 0.f, 0.2f, 0.f );

				// Draw the trunk
				//drawCylinder( treeData.position, treeData.trunkDiameter/2.f, 
				//	treeData.height-treeData.crownHeight, brown );

				// Draw the lower part of the crown
				vector3D crownPos = treeData.position;
				crownPos.z += treeData.height-treeData.crownHeight;
				drawCylinder( crownPos, treeData.crownRadius, 
					treeData.crownHeight-treeData.coneHeight, green );

				// Draw Upper part of the crown
				vector3D apex = vector3D( crownPos.x, crownPos.y, treeData.position.z+treeData.height);
				drawCone( apex, treeData.crownRadius, treeData.coneHeight, green );

			} // end draw rocket
			else if( treeData.shape == TreeType::Shape::Cone ){
				vector3D green = vector3D( 0.f, 0.3f, 0.f );

				float alpha = 0.1f;

				// Draw the trunk
				//drawCylinder( treeData.position, treeData.trunkDiameter/2.f, 
				//	treeData.height-treeData.crownHeight, brown );

				// Draw the lower part of the crown
				vector3D crownPos = treeData.position;
				crownPos.z += treeData.height-treeData.crownHeight;
				drawCylinder( crownPos, treeData.crownRadius, alpha, green );

				// Draw Upper part of the crown
				vector3D apex = vector3D( crownPos.x, crownPos.y, treeData.position.z+treeData.height);
				drawCone( apex, treeData.crownRadius, treeData.coneHeight-alpha, green );

			} // end draw cone
			else if( treeData.shape == TreeType::Shape::Ellipsoid ){
				vector3D green = vector3D( 0.f, 0.4f, 0.f );

				// Draw the trunk
				//drawCylinder( treeData.position, treeData.trunkDiameter/2.f, 
				//	treeData.height-treeData.crownHeight, brown );

				// Draw the crown
				vector3D center = treeData.position;
				center.z += (treeData.height-treeData.crownHeight) + (treeData.crownHeight/2.f);
				vector3D radii = vector3D( treeData.crownRadius, treeData.crownRadius, treeData.crownHeight/2.f );
				drawEllipse( center, radii, green );

			} // end draw ellipsoid
			else if( treeData.shape == TreeType::Shape::Canopy ){
				vector4D green = vector4D( 0.f, 0.4f, 0.f, 1.f );

				// Draw the trunk
				//drawCylinder( treeData.position, treeData.trunkDiameter/2.f, 
				//	treeData.height-treeData.crownHeight, brown );

				// Draw the crown
				vector3D boxmax = vector3D(	treeData.position.x+treeData.length,
								treeData.position.y+treeData.width,
								treeData.position.z+treeData.height );
				drawBox( treeData.position, boxmax, green );

			} // end draw ellipsoid 

		} break; // end draw tree

		case VolumeType::Box:{
			vector4D color = vector4D( renderColor.x, renderColor.y, renderColor.z, 1.f );
			drawBox( air_boxmin, air_boxmax, color );
		} break;

	} // end switch volume type
} // end draw volume


// http://davidwparker.com/2011/09/05/opengl-screencast-8-drawing-in-3d-part-3-spheres/
void VolumeCell::drawCylinder( vector3D pos, float radius, float height, vector3D color ){

	//  sides
	glPushMatrix();
	glTranslatef( pos.x, pos.y, pos.z+height/2.f );
	glRotatef( 90.f, 1.f, 0.f, 0.f );
	glBegin( GL_QUAD_STRIP );
	glColor3f(color.x,color.y,color.z);
	for( int j = 0; j <= 360; j += DegreesRot ){
		glVertex3f( radius*Cos(j), +height/2.f, radius*Sin(j) );
		glVertex3f( radius*Cos(j), -height/2.f, radius*Sin(j) );
	}
	glEnd();
/*
	// top and bottom circles
	// reuse the currentTexture on top and bottom)
	for( int i = height/2.f; i >= -height/2.f; i -= height ) {

		glBegin(GL_TRIANGLE_FAN);
		glColor3f(0.0,0.0,1.0);
		glVertex3f(0,i,0);
		for (int k=0;k<=360;k+=DegreesRot) {
			glColor3f(1.0,0.0,0.0);
			glVertex3f( Cos(k), i, Sin(k) );
		}
		glEnd();
	}
*/

	glPopMatrix();

} // end draw cylinder


// http://davidwparker.com/2011/09/05/opengl-screencast-8-drawing-in-3d-part-3-spheres/
void VolumeCell::drawCone( vector3D apex, float radius, float height, vector3D color ){

	glPushMatrix();
	glTranslatef( apex.x, apex.y, apex.z );

	// sides 
	glBegin(GL_TRIANGLES);
	glColor3f(color.x,color.y,color.z);
	for (int k=0;k<=360;k+=DegreesRot){
		glVertex3f(0,0,1);
		glVertex3f(radius*Cos(k),radius*Sin(k),-height);
		glVertex3f(radius*Cos(k+DegreesRot),radius*Sin(k+DegreesRot),-height);
	}
	glEnd();

/*
	// bottom circle  
	// rotate back 
	glRotated(90,1,0,0);
	glBegin(GL_TRIANGLES);
	for (int k=0;k<=360;k+=DegreesRot) {
		glColor3f(1.0,0.0,0.0);
		glVertex3f(0,-height,0);
		glColor3f(1.0,0.0,1.0);
		glVertex3f(radius*Cos(k),-height,radius*Sin(k));
		glColor3f(1.0,1.0,0.0);
		glVertex3f(radius*Cos(k+DegreesRot),-height,radius*Sin(k+DegreesRot));
	}
	glEnd();
*/
	glPopMatrix();

} // end draw cone


void VolumeCell::drawEllipse( vector3D center, vector3D radii, vector3D color ){

	glPushMatrix();
	glTranslatef( center.x, center.y, center.z );
	glColor3f(color.x,color.y,color.z);

	GLUquadricObj* obj = gluNewQuadric();
	glScalef(radii.x, radii.y, radii.z);
	gluSphere(obj, 1.0, 10, 10);
	gluDeleteQuadric(obj);

	glPopMatrix();

} // end draw ellipse



void VolumeCell::drawBox( vector3D boxmin, vector3D boxmax, vector4D color ){


	float nx = (boxmax.x - boxmin.x)/2.f;
	float ny = (boxmax.y - boxmin.y)/2.f;
	float nz = (boxmax.z - boxmin.z)/2.f;
	float x = boxmin.x + nx;
	float y = boxmin.y + ny;
	float z = boxmin.z + nz;
	float alpha = 0.001f;

	glPushMatrix();

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glColor4f(color.x,color.y,color.z,color.w);
	glBegin(GL_QUADS);
	{
		// Top
		glNormal3f( 0.f, 0.f, 1.f );
		glVertex3f(x - nx, y - ny, z + nz + alpha);	// A
		glVertex3f(x - nx, y + ny, z + nz + alpha);	// B
		glVertex3f(x + nx, y + ny, z + nz + alpha);	// C
		glVertex3f(x + nx, y - ny, z + nz + alpha);	// D

		// Bottom
		glNormal3f( 0.f, 0.f, -1.f );
		glVertex3f(x - nx, y - ny, z - nz + alpha);	// E
		glVertex3f(x + nx, y - ny, z - nz + alpha);	// H
		glVertex3f(x + nx, y + ny, z - nz + alpha);	// G
		glVertex3f(x - nx, y + ny, z - nz + alpha);	// F

		// Front
		glNormal3f( 0.f, 1.f, 0.f );
		glVertex3f(x - nx, y - ny, z + nz + alpha);	// A
		glVertex3f(x + nx, y - ny, z + nz + alpha);	// D
		glVertex3f(x + nx, y - ny, z - nz + alpha);	// H
		glVertex3f(x - nx, y - ny, z - nz + alpha);	// E

		// Back
		glNormal3f( 0.f, -1.f, 0.f );
		glVertex3f(x + nx, y + ny, z + nz + alpha);	// C
		glVertex3f(x - nx, y + ny, z + nz + alpha);	// B
		glVertex3f(x - nx, y + ny, z - nz + alpha);	// F
		glVertex3f(x + nx, y + ny, z - nz + alpha);	// G

		// Left
		glNormal3f( 0.f, 1.f, 0.f );
		glVertex3f(x - nx, y + ny, z + nz + alpha);	// B
		glVertex3f(x - nx, y - ny, z + nz + alpha);	// A
		glVertex3f(x - nx, y - ny, z - nz + alpha);	// E
		glVertex3f(x - nx, y + ny, z - nz + alpha);	// F

		// Right
		glNormal3f( 0.f, -1.f, 0.f );
		glVertex3f(x + nx, y - ny, z + nz + alpha);	// D
		glVertex3f(x + nx, y + ny, z + nz + alpha);	// C
		glVertex3f(x + nx, y + ny, z - nz + alpha);	// G
		glVertex3f(x + nx, y - ny, z - nz + alpha);	// H

	}
	glEnd();

	glDisable(GL_BLEND);

	glPopMatrix();



} // end draw box








































void BoxCell::draw(){


	float nx = (boxmax.x - boxmin.x)/2.f;
	float ny = (boxmax.y - boxmin.y)/2.f;
	float nz = (boxmax.z - boxmin.z)/2.f;
	float x = boxmin.x + nx;
	float y = boxmin.y + ny;
	float z = boxmin.z + nz;
	float alpha = 0.001f;

	glPushMatrix();

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glColor4f( color.x, color.y, color.z, color.w );
	glBegin(GL_QUADS);
	{
		// Top
		glNormal3f( 0.f, 0.f, 1.f );
		glVertex3f(x - nx, y - ny, z + nz + alpha);	// A
		glVertex3f(x - nx, y + ny, z + nz + alpha);	// B
		glVertex3f(x + nx, y + ny, z + nz + alpha);	// C
		glVertex3f(x + nx, y - ny, z + nz + alpha);	// D

		// Bottom
		glNormal3f( 0.f, 0.f, -1.f );
		glVertex3f(x - nx, y - ny, z - nz + alpha);	// E
		glVertex3f(x + nx, y - ny, z - nz + alpha);	// H
		glVertex3f(x + nx, y + ny, z - nz + alpha);	// G
		glVertex3f(x - nx, y + ny, z - nz + alpha);	// F

		// Front
		glNormal3f( 0.f, 1.f, 0.f );
		glVertex3f(x - nx, y - ny, z + nz + alpha);	// A
		glVertex3f(x + nx, y - ny, z + nz + alpha);	// D
		glVertex3f(x + nx, y - ny, z - nz + alpha);	// H
		glVertex3f(x - nx, y - ny, z - nz + alpha);	// E

		// Back
		glNormal3f( 0.f, -1.f, 0.f );
		glVertex3f(x + nx, y + ny, z + nz + alpha);	// C
		glVertex3f(x - nx, y + ny, z + nz + alpha);	// B
		glVertex3f(x - nx, y + ny, z - nz + alpha);	// F
		glVertex3f(x + nx, y + ny, z - nz + alpha);	// G

		// Left
		glNormal3f( 0.f, 1.f, 0.f );
		glVertex3f(x - nx, y + ny, z + nz + alpha);	// B
		glVertex3f(x - nx, y - ny, z + nz + alpha);	// A
		glVertex3f(x - nx, y - ny, z - nz + alpha);	// E
		glVertex3f(x - nx, y + ny, z - nz + alpha);	// F

		// Right
		glNormal3f( 0.f, -1.f, 0.f );
		glVertex3f(x + nx, y - ny, z + nz + alpha);	// D
		glVertex3f(x + nx, y + ny, z + nz + alpha);	// C
		glVertex3f(x + nx, y + ny, z - nz + alpha);	// G
		glVertex3f(x + nx, y - ny, z - nz + alpha);	// H

	}
	glEnd();

	glDisable(GL_BLEND);

	glPopMatrix();

}





