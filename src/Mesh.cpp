#include "Mesh.h"

float Mesh::getHeight(float x, float y)
{
	return tris->heightToTri(x,y);
}
