/* File: VolumeCell.h
 * Author: Matthew Overby
 *
 * THIS CLASS IS FOR TESTING PURPOSES ONLY
 * Eventually most/all of these functions and parameters will
 * be replaced by something more efficient.
 *
 * This class is just a placeholder for visualizing volumes
 */

#ifndef SLUI_VOLUMECELL_H
#define SLUI_VOLUMECELL_H

#include "vector3D.h"
#include "vector4D.h"
#include <SFML/Graphics.hpp>
#include <math.h>

#define Cos(theta) cos(3.14159/180*(theta))
#define Sin(theta) sin(3.14159/180*(theta))
#define DegreesRot 5

namespace SLUI {

#ifndef TreeTypeNS
#define TreeTypeNS
namespace TreeType {

	namespace Shape {
		static const int Cone = 1;
		static const int Ellipsoid = 2;
		static const int Rocket = 3;
		static const int Canopy = 4;
	}

	namespace Foliage {
		static const int Trunk = 0;
		static const int Horizontal = 1;
		static const int Vertical = 2;
		static const int Isotropic = 3;
		static const int Conical = 4;
		static const int Ellipsoidal = 5;
	}
}
#endif

namespace VolumeType {

	static const int Tree = 1;
	static const int Box = 2;
}

struct TreeCell {

	int shape; // Crown shape
	vector3D position;
	float height;
	float crownHeight;
	float crownRadius;
	float trunkDiameter;
	float coneHeight; // For rocket shaped trees
	float width, length; // For canopy shaped trees
};

class VolumeCell {

	public:
		VolumeCell();
		VolumeCell( struct TreeCell tc );
		void draw();
		~VolumeCell();

		int volumeType;
		bool renderable;

		// FOR AIRCELLS
		vector3D air_boxmin;
		vector3D air_boxmax;
		int layer;
		vector3D renderColor;

	private:
		void drawCylinder( vector3D pos, float radius, float height, vector3D color );
		void drawCone( vector3D apex, float radius, float height, vector3D color );
		void drawEllipse( vector3D center, vector3D radii, vector3D color );
		void drawBox( vector3D boxmin, vector3D boxmax, vector4D color );
		TreeCell treeData;

};

class RenderableCell {

	public:
		virtual void draw() = 0;
		virtual void setColor( vector4D newColor ) = 0;
		virtual vector3D getPosition() = 0;
};

class DataCell {
	public:
};

class BoxCell : public RenderableCell {
	public:
		BoxCell( vector3D min, vector3D max ) : 
			boxmin( min ), boxmax( max ) { color = vector4D( 1.f, 0.f, 0.f, 1.f ); }
		void setColor( vector4D newColor ){
			color = newColor;
		//	if( color.x > 1.f || color.y > 1.f || color.z > 1.f ){
		//		color.x /= 255.f;
		//		color.y /= 255.f;
		//		color.z /= 255.f;
		//		color.w /= 255.f;
		//	}
		}
		vector3D getPosition(){
			float x = ( boxmax.x + boxmin.x ) / 2.f;
			float y = ( boxmax.y + boxmin.y ) / 2.f;
			float z = ( boxmax.z + boxmin.z ) / 2.f;
			vector3D result = vector3D( x, y, z );
			return result;
		}
		void draw();

	private:
		vector3D boxmin;
		vector3D boxmax;
		vector4D color;
};

typedef std::map<unsigned int,RenderableCell*> RenderMap;

}

#endif

