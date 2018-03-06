/* File: CellTracker.cpp
 * Author: Matthew Overby
 */

#include <GL/glew.h>
#include "CellTracker.h"


using namespace SLUI;

CellTracker::CellTracker(EventTracker* evT, sf::RenderWindow* app){

	m_App = app;
	m_eventTracker = evT;

} // end constructor


CellTracker::~CellTracker(){

	RenderMap::iterator it;

	it = m_renderCells.begin();
	while( it != m_renderCells.end() ){
		delete it->second;
		it++;
	}

	cellList.clear();

} // end destructor


void CellTracker::addVolume( unsigned int id, VolumeCell vc ){

	volumeList[id] = vc;

} // end add volume to volume list


void CellTracker::addPatch( unsigned int id, int face, vector3D anchor, vector3D v1, vector3D v2 ){

	Cell newCell;

	newCell.id = id;
	newCell.highlightColor = vector3D(1,0,0);
	newCell.type = CellType::Patch;
	newCell.face = face;

	std::stringstream cubeStats;
	cubeStats << "Patch ID: " << id << "\n";
	newCell.stats.push_back( cubeStats.str() );

	newCell.anchor = anchor;
	newCell.v1 = v1;
	newCell.v2 = v2;
	newCell.normal = v1 % v2;

	float alpha = 0.01f;
	if( face == CellType::UP_FACE ){
		newCell.a = anchor;
		newCell.b = vector3D( anchor.x+v2.x, anchor.y+v2.y, anchor.z+v2.z + alpha );
		newCell.c = vector3D( anchor.x+v2.x+v1.x, anchor.y+v2.y+v1.y, anchor.z+v2.z+v1.z + alpha );
		newCell.d = vector3D( anchor.x+v1.x, anchor.y+v1.y, anchor.z+v1.z + alpha );
	}
	else if( face == CellType::DOWN_FACE ){
		newCell.a = anchor;
		newCell.b = vector3D( anchor.x+v2.x, anchor.y+v2.y, anchor.z+v2.z - alpha );
		newCell.c = vector3D( anchor.x+v2.x+v1.x, anchor.y+v2.y+v1.y, anchor.z+v2.z+v1.z - alpha );
		newCell.d = vector3D( anchor.x+v1.x, anchor.y+v1.y, anchor.z+v1.z - alpha );
	}
	else if( face == CellType::NORTH_FACE ){
		newCell.a = anchor;
		newCell.b = vector3D( anchor.x+v2.x, anchor.y+v2.y + alpha, anchor.z+v2.z );
		newCell.c = vector3D( anchor.x+v2.x+v1.x, anchor.y+v2.y+v1.y + alpha, anchor.z+v2.z+v1.z );
		newCell.d = vector3D( anchor.x+v1.x, anchor.y+v1.y + alpha, anchor.z+v1.z );
	}
	else if( face == CellType::SOUTH_FACE ){
		newCell.a = anchor;
		newCell.b = vector3D( anchor.x+v2.x, anchor.y+v2.y - alpha, anchor.z+v2.z );
		newCell.c = vector3D( anchor.x+v2.x+v1.x, anchor.y+v2.y+v1.y - alpha, anchor.z+v2.z+v1.z );
		newCell.d = vector3D( anchor.x+v1.x, anchor.y+v1.y - alpha, anchor.z+v1.z );
	}
	else if( face == CellType::EAST_FACE ){
		newCell.a = anchor;
		newCell.b = vector3D( anchor.x+v2.x + alpha, anchor.y+v2.y, anchor.z+v2.z );
		newCell.c = vector3D( anchor.x+v2.x+v1.x + alpha, anchor.y+v2.y+v1.y, anchor.z+v2.z+v1.z );
		newCell.d = vector3D( anchor.x+v1.x + alpha, anchor.y+v1.y, anchor.z+v1.z );
	}
	else if( face == CellType::WEST_FACE ){
		newCell.a = anchor;
		newCell.b = vector3D( anchor.x+v2.x - alpha, anchor.y+v2.y, anchor.z+v2.z );
		newCell.c = vector3D( anchor.x+v2.x+v1.x - alpha, anchor.y+v2.y+v1.y, anchor.z+v2.z+v1.z );
		newCell.d = vector3D( anchor.x+v1.x - alpha, anchor.y+v1.y, anchor.z+v1.z );
	}

	float v1_dot_v1 = v1 * v1;
	float v2_dot_v2 = v2 * v2;
	v1_dot_v1 = 1.f / v1_dot_v1;
	v2_dot_v2 = 1.f / v2_dot_v2;
	newCell.v1 = v1 * v1_dot_v1;
	newCell.v2 = v2 * v2_dot_v2;

	cellList.insert( std::pair<unsigned int, Cell>( id, newCell ) );

} // end add patch to cell list


void CellTracker::setCellColor( unsigned int id, vector4D newColor ){

	if(newColor.x > 1.f) newColor.x = newColor.x/255.f;
	if(newColor.y > 1.f) newColor.y = newColor.y/255.f;
	if(newColor.z > 1.f) newColor.z = newColor.z/255.f;
	if(newColor.w > 1.f) newColor.w = newColor.w/255.f;

	RenderMap::iterator it = m_renderCells.find( id );
	if( it != m_renderCells.end() ){
		it->second->setColor( newColor );
	}
} // end set cell render color


void CellTracker::setCellColor( unsigned int id, vector3D newColor ){

	if(newColor.x > 1) newColor.x = newColor.x/255;
	if(newColor.y > 1) newColor.y = newColor.y/255;
	if(newColor.z > 1) newColor.z = newColor.z/255;

	std::map<unsigned int,Cell>::iterator it = cellList.find( id );
	if( it != cellList.end() ){
		it->second.renderColor = newColor;
	}
} // end set cell render color

void CellTracker::setCellHighlightColor( unsigned int id, vector3D newColor ){

	if(newColor.x > 1) newColor.x = newColor.x/255;
	if(newColor.y > 1) newColor.y = newColor.y/255;
	if(newColor.z > 1) newColor.z = newColor.z/255;

	std::map<unsigned int,Cell>::iterator it = cellList.find( id );
	if( it != cellList.end() ){
		it->second.highlightColor = newColor;
	}
} // end set highlight color


void CellTracker::addStat( unsigned int id, std::string stat ){

	std::map<unsigned int,Cell>::iterator it = cellList.find( id );
	if( it != cellList.end() ){
		it->second.stats.push_back( stat );
	}
} // end add stat


void CellTracker::clearStat( unsigned int id){
	
	std::map<unsigned int,Cell>::iterator it = cellList.find( id );
	if( it != cellList.end() ){
		it->second.stats.clear();
	}
} // end clea stat


std::string CellTracker::getStats( unsigned int id ){

	std::map<unsigned int,Cell>::iterator it = cellList.find( id );
	if( it != cellList.end() ){
		std::stringstream stats("");
		for(int i=0; i < it->second.stats.size(); i++){
			stats << it->second.stats.at(i) << "\n";
		}
		return stats.str();
	}

	return "";
} // end get stats


void CellTracker::setVolumeRenderColor( unsigned int id, vector3D color ){

	std::map<unsigned int, VolumeCell>::iterator finder = volumeList.find( id );
	if( finder != volumeList.end() ){
		finder->second.renderColor = color;
	}
	else{
		//std::cout << "setVolumeRenderColor: Could not find Volume " << id << std::endl;
	}
}


void CellTracker::drawTrees(){

	std::map<unsigned int, VolumeCell>::iterator it;
	for( it = volumeList.begin(); it != volumeList.end(); it++ ){
		if( it->second.volumeType == VolumeType::Tree ){
			vector3D renderColor = it->second.renderColor;
			//vector3D renderColor = vector3D( 0.5f, 0.5f, 0.5f );
			float alpha = 1.f; // it->second.attenuation
			glColor4f( (GLfloat)renderColor.x, (GLfloat)renderColor.y, (GLfloat)renderColor.z, alpha );
			it->second.draw();
		}
	} // end loop through volumes

} // end draw volumes


void CellTracker::drawAircells( int layer ){

	if( layer > 0 ){
		glEnable(GL_DEPTH_TEST);
		std::map<unsigned int, VolumeCell>::iterator it;
		for( it = volumeList.begin(); it != volumeList.end(); it++ ){
			if( ( layer == it->second.layer ) && it->second.volumeType == VolumeType::Box ){
				vector3D renderColor = it->second.renderColor;
				//vector3D renderColor = vector3D( 0.5f, 0.5f, 0.5f );
				float alpha = 1.f; // it->second.attenuation
				glColor3f( (GLfloat)renderColor.x, (GLfloat)renderColor.y, (GLfloat)renderColor.z );
				it->second.draw();
			}
		} // end loop through volumes
		glDisable(GL_DEPTH_TEST);
	}
} // end draw volumes


void CellTracker::drawSelected(){

	for(int i=0; i<selectedCells.size(); i++){
		int id = selectedCells.at(i);
		std::map<unsigned int, Cell>::iterator it = cellList.find( id );
		if( it != cellList.end() ){

		if( it->second.type == CellType::Patch ){

			glPushMatrix();
			//glEnable(GL_BLEND);
			//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			glColor3f(it->second.highlightColor.x, it->second.highlightColor.y, it->second.highlightColor.z);
			float alpha = 0.001;

			glBegin(GL_QUADS);
			{
				glVertex3f(it->second.a.x, it->second.a.y, it->second.a.z);
				glVertex3f(it->second.b.x, it->second.b.y, it->second.b.z);
				glVertex3f(it->second.c.x, it->second.c.y, it->second.c.z);
				glVertex3f(it->second.d.x, it->second.d.y, it->second.d.z);
			}
			glEnd();
			glPopMatrix();

		} // end if type
			
		} // end if cell if found in cellList

	} // end loop through cellList
} // end draw selected


void CellTracker::checkIntersect(){

	const sf::Input &input = m_App->GetInput();
	int mouse_x = input.GetMouseX();
	int mouse_y = input.GetMouseY();
	GLint viewport[4];
	GLdouble modelMatrix[16];
	GLdouble projectionMatrix[16];
	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, projectionMatrix);
	GLfloat winY = GLfloat(viewport[3] - mouse_y);

	clickRay ray;
	double x, y, z;

	// Near
	gluUnProject( (double)mouse_x, winY, 0.0f, modelMatrix, projectionMatrix, viewport, &x, &y, &z );
	ray.origin = vector3D( x, y, z );

	// Far
	gluUnProject( (double)mouse_x, winY, 1.0f, modelMatrix, projectionMatrix, viewport, &x, &y, &z );
	ray.direction = vector3D( x - ray.origin.x, y - ray.origin.y, z - ray.origin.z );

	std::map< unsigned int, Cell >::iterator it;
	for( it = cellList.begin(); it != cellList.end(); it++){
		it->second.intersect( &ray );
	} // end loop through cellList

	if( ray.hitID >= 0 ){
		it = cellList.find( ray.hitID );
		if( it != cellList.end() ){
			if( !it->second.selected ){
				it->second.selected = true;
				selectedCells.push_back( it->first );
				m_eventTracker->callFunction( "cellselected" );
			}
			else{
				removeSelected( it->first );
			}
		} // end cell id valid
	} // end a cell was hit

} // end check intersect


void CellTracker::checkIntersect( clickRay &ray ){

	std::map< unsigned int, Cell >::iterator it;
	for( it = cellList.begin(); it != cellList.end(); it++){
		it->second.intersect( &ray );
	} // end loop through cellList

	if( ray.hitID >= 0 ){
		it = cellList.find( ray.hitID );
		if( it != cellList.end() ){
			if( !it->second.selected ){
				it->second.selected = true;
				selectedCells.push_back( it->first );
				m_eventTracker->callFunction( "cellselected" );
			}
			else{
				removeSelected( it->first );
			}
		} // end cell id valid
	} // end a cell was hit

} // end check intersect


void CellTracker::checkEvent(){

	const sf::Input &input = m_App->GetInput();

	switch(m_eventTracker->eventType){

		case EventType::M_Clickleft:{
			if(input.IsKeyDown(sf::Key::LShift)){
				checkIntersect();
			}
		} break;

		default:{
		} break;
	}
}


void CellTracker::clearCellList(){

	cellList.clear();
	selectedCells.clear();
}

void CellTracker::clearSelected(){

	while( !selectedCells.empty() ){
		int id = selectedCells.back();
		selectedCells.pop_back();
		std::map<unsigned int,Cell>::iterator it = cellList.find( id );
		if( it != cellList.end() ){
			it->second.selected = false;
			m_eventTracker->showMessage("", 0);
		}
	}
}

void CellTracker::setSelected( unsigned int id ){

	// remove it if it's already there and push it on top of the stack
	for(int i=0; i<selectedCells.size(); i++){
		if( id == selectedCells.at(i) ){
			selectedCells.erase( selectedCells.begin()+i );
		}
	}
	selectedCells.push_back(id);
}

void CellTracker::removeCell( unsigned int id ){

	std::map<unsigned int,Cell>::iterator it = cellList.find( id );
	if( it != cellList.end() ){
		cellList.erase( it );
	}
}

void CellTracker::removeSelected( unsigned int id ){

	std::map<unsigned int,Cell>::iterator it = cellList.find( id );
	if( it != cellList.end() ){
		it->second.selected = false;
	}

	for(int i=0; i<selectedCells.size(); i++){
		if( id == selectedCells.at(i) ){
			selectedCells.erase( selectedCells.begin()+i );
		}
	}
}

void CellTracker::testPrintCells(){

	std::map<unsigned int, Cell>::iterator it;
	for (it = cellList.begin(); it != cellList.end(); it++) {
		std::cout << "Cell ID: " << it->first << std::endl;
	}
}


/*
*	CELL/PATCHES
*/


clickRay::clickRay(){

	origin = vector3D( 0.f, 0.f, 0.f );
	direction = vector3D( 0.f, 0.f, 0.f );
	closest = tmax;
	hitID = -1;
}


Cell::Cell(){

	selected = false;
	id = -1;

	renderColor = vector3D(0,0,0);
	a = vector3D(0,0,0);
	b = vector3D(0,0,0);
	c = vector3D(0,0,0);
	d = vector3D(0,0,0);

}


void Cell::intersect( clickRay *ray ){

	if( type == CellType::Patch ){

		float faceW = normal * anchor;
		float dt = ray->direction * normal;
		float n_dot_rayorigin = normal * ray->origin;
		float t = ( faceW - n_dot_rayorigin ) / dt;
		if( t > ray->tmin && t < ray->tmax ){
			vector3D rayDir_times_t = ray->direction * t;
			vector3D p = ray->origin + rayDir_times_t;
			vector3D vi = p - anchor;
			float a1 = v1 * vi;
			if( a1 >= 0 && a1 <= 1 ){
				float a2 = v2 * vi;
				if( a2 >= 0 && a2 <= 1 ){
					if( t > 0.f && t < ray->closest ){
						ray->closest = t;
						ray->hitID = id;
					} // a closest hit
				}
			}
		} // end t within bounds

	} // end check patch intersection

} // end check intersection


void CellTracker::renderCells(){

	RenderMap::iterator cell = m_renderCells.begin();
	while( cell != m_renderCells.end() ){
//			glPushMatrix();
		cell->second->draw();
//				glFlush();
//			glPopMatrix();
		cell++;
	}
}


bool CellTracker::addRenderableCell( unsigned int id, RenderableCell *cell ){

	bool addSuccess = false;
	int size = m_renderCells.size();
	m_renderCells.insert( std::pair<unsigned int,RenderableCell*>( id, cell ) );
	if( m_renderCells.size() > size ){ addSuccess = true; }
	return addSuccess;

}


RenderableCell* CellTracker::getRenderableCell( unsigned int id ){

	RenderableCell *r_cell = 0;
	RenderMap::iterator cell = m_renderCells.find( id );
	if( cell != m_renderCells.end() ){
		r_cell = cell->second;
	}

	return r_cell;

}


// These are static values in a struct, and thus, must be defined in an object
const float clickRay::tmin = -1.f;
const float clickRay::tmax = 1.f;
