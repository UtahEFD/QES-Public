/* File: Camera.cpp
 * Author: Matthew Overby
 */

#include "Camera.h"

using namespace SLUI;

Camera::Camera(vector3D pos, vector3D lap, vector3D u, OptionTracker *opt, EventTracker *evt, sf::RenderWindow *app) {

	position = pos;
	lookAtPoint = lap;
	direction = lookAtPoint - position;
	m_eventTracker = evt;
	m_optTracker = opt;
	m_App = app;

	up = u;
	createOrth(up);

	default_position = position;
	default_lookAtPoint = lookAtPoint;
	default_up = up;

	orthoMove = vector3D( 0.f, 0.f, 1.f );

	farPlane = 500.f;
	scale = 5.f;

} // end constructor

void Camera::checkEvent() {

	float m_sens = m_optTracker->getValue("mousespeed");
	int invert = 1;
	if( m_optTracker->getActive("minvert") ) invert = -1;
	int initX = m_eventTracker->oldPos.x;
	int initY = m_eventTracker->oldPos.y;

	if( m_optTracker->stateChanged("camera.view") ){ setToDefault(); }

	const sf::Input &input = m_App->GetInput();
	if( input.IsKeyDown( sf::Key::LShift ) ){ return; }

	switch( m_eventTracker->eventType ){

		case EventType::M_Scrollup:{

			float w_amount = -1.5f * m_sens;
			std::string cameraView = m_optTracker->getListValue( "camera.view" );
			if( cameraView.compare("Free Move") == 0 || cameraView.compare("Model View") == 0 ){
				if(position.distance(lookAtPoint) > 2) move(0, 0, w_amount);
			}
			else {
				float oldZ = orthoMove.z;
				if( oldZ + w_amount*0.05 < 0.1f + w_amount*0.05 ){ orthoMove.z = 0.1f; }
				else { orthoMove.z += w_amount*0.05; }
			}

		} break;

		case EventType::M_Scrolldown:{

			float w_amount = 1.5f * m_sens;
			std::string cameraView = m_optTracker->getListValue("camera.view");
			if( cameraView.compare("Free Move") == 0 || cameraView.compare("Model View") == 0 ){
				move(0, 0, w_amount);
			}
			else {
				orthoMove.z += w_amount*0.05;
			}

		} break;

		case EventType::M_Dragleft:{

			std::string cameraView = m_optTracker->getListValue("camera.view");
			if( cameraView.compare("Free Move") == 0 ){

				float vert = (initY - m_eventTracker->newPos.y) * 0.0035 * m_sens * invert;
				float horz = (initX - m_eventTracker->newPos.x) * 0.0035 * m_sens;
				look(vert, horz);

				initX = m_eventTracker->oldPos.x;
				initY = m_eventTracker->oldPos.y;
			}
			else if( cameraView.compare("Model View") == 0 ){

				float vert = (initY - m_eventTracker->newPos.y) * 0.0035 * m_sens * invert;
				float horz = (initX - m_eventTracker->newPos.x) * 0.0035 * m_sens;
				float dist = position.distance(lookAtPoint);

				move(0, 0, -dist);
				look(vert, horz);
				move(0, 0, dist);

				initX = m_eventTracker->oldPos.x;
				initY = m_eventTracker->oldPos.y;
			}
			else {

				float vert = (initY - m_eventTracker->newPos.y) * 0.035 * m_sens;
				float horz = (initX - m_eventTracker->newPos.x) * 0.035 * m_sens;
				float dist = position.distance(lookAtPoint);

				orthoMove.x -= horz;
				orthoMove.y -= vert;

				initX = m_eventTracker->oldPos.x;
				initY = m_eventTracker->oldPos.y;
			}

		} break;

		case EventType::M_Dragright:{

			std::string cameraView = m_optTracker->getListValue("camera.view");
			if( cameraView.compare("Free Move") == 0 ){

				float v_amount = (initY - m_eventTracker->newPos.y) * 0.05 * m_sens;
				float u_amount = (initX - m_eventTracker->newPos.x) * -0.05 * m_sens;
				move(u_amount, v_amount, 0);

				initX = m_eventTracker->oldPos.x;
				initY = m_eventTracker->oldPos.y;
			}
			else if( cameraView.compare("Model View") == 0 ){

				float v_amount = (initY - m_eventTracker->newPos.y) * 0.05 * m_sens;
				float u_amount = (initX - m_eventTracker->newPos.x) * -0.05 * m_sens;
				vector3D posBefore = position;
				move(u_amount, v_amount, 0);
				vector3D posAfter = position;
				vector3D diff = (posAfter - posBefore);
				lookAtPoint = lookAtPoint + diff;

				initX = m_eventTracker->oldPos.x;
				initY = m_eventTracker->oldPos.y;
			}
			else {

				float v_amount = (initY - m_eventTracker->newPos.y) * 0.0025 * m_sens;
				float oldZ = orthoMove.z;
				if( oldZ - v_amount < 0.1f + v_amount ){
					orthoMove.z = 0.1f;
				}
				else {
					orthoMove.z -= v_amount;
				}
			}

		} // end drag mouse right button

		case -1:{
		} break;

	} // end switch event type

} // end check event

void Camera::setGLView(){

	glViewport(0, 0, m_App->GetWidth(), m_App->GetHeight());

	std::string cameraView = m_optTracker->getListValue("camera.view");
	if( cameraView.compare("Free Move") == 0 || cameraView.compare("Model View") == 0 ){

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		
		// Need to compute the aspect ratio so the view will be correct with square pixels
		float aspectRatio = m_App->GetWidth() / (float)m_App->GetHeight(); 

		gluPerspective(90.f, aspectRatio, 1.f, farPlane);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		gluLookAt(	position.x,	position.y,	position.z,
				position.x - wVec.x,		position.y - wVec.y,	position.z - wVec.z,
				up.x,		up.y,		up.z	);

	}
	else {

		float zoom = orthoMove.z;
		float xMove = orthoMove.x * scale;
		float yMove = orthoMove.y * scale;
		float aspectRatio = (float)m_App->GetWidth() / (float)m_App->GetHeight();

		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();

		if( cameraView.compare("Top-Down") == 0 ){

			glRotatef( 180.f, 0.f, 0.f, 1.f );
			glOrtho(	(20*scale)*zoom*aspectRatio-xMove,	-(20*scale)*zoom*aspectRatio-xMove, 
					(20*scale)*zoom+yMove,			-(20*scale)*zoom+yMove, 
					(-20*scale),				(20*scale)	);
		}
		else if( cameraView.compare("North Face") == 0 ){

			glRotatef( 90.f, 1.f, 0.f, 0.f );
			glOrtho(	(20*scale)*zoom*aspectRatio+xMove,	-(20*scale)*zoom*aspectRatio+xMove, 
					(20*scale),				-(20*scale), 
					(-20*scale)*zoom-yMove,			(20*scale)*zoom-yMove	);
		}
		else if( cameraView.compare("South Face") == 0 ){

			glRotatef( 90.f, 1.f, 0.f, 0.f );
			glRotatef( 180.f, 0.f, 0.f, 1.f );
			glOrtho(	(20*scale)*zoom*aspectRatio-xMove,	-(20*scale)*zoom*aspectRatio-xMove, 
					(20*scale),				-(20*scale), 
					(-20*scale)*zoom-yMove,			(20*scale)*zoom-yMove	);
		}
		else if( cameraView.compare("East Face") == 0 ){

			glRotatef( 90.f, 1.f, 0.f, 0.f );
			glRotatef( 90.f, 0.f, 0.f, 1.f );
			glOrtho(	(20*scale),				-(20*scale), 
					(20*scale)*zoom*aspectRatio-xMove,	-(20*scale)*zoom*aspectRatio-xMove, 
					(-20*scale)*zoom-yMove,			(20*scale)*zoom-yMove	);
		}
		else if( cameraView.compare("West Face") == 0 ){

			glRotatef( 90.f, 1.f, 0.f, 0.f );
			glRotatef( 270.f, 0.f, 0.f, 1.f );
			glOrtho(	(20*scale),				-(20*scale), 
					(20*scale)*zoom*aspectRatio+xMove,	-(20*scale)*zoom*aspectRatio+xMove, 
					(-20*scale)*zoom-yMove,			(20*scale)*zoom-yMove	);
		}

		// glOrtho(left,right,top,bottom,near,far);

		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();

	}

} // end set gl view

void Camera::setLookat(vector3D newLookat){

	// If the position is equal to the lookat, we'll run in
	// to obvious problems when we calculate the view direction.
	// So, increment lookat if this is the case
	lookAtPoint = newLookat;
	if( lookAtPoint.x == position.x ) lookAtPoint.x++;
	if( lookAtPoint.y == position.y ) lookAtPoint.y++;
	if( lookAtPoint.z == position.z ) lookAtPoint.z++;

	direction = lookAtPoint - position;
	createOrth(up);

} // end set look at point

void Camera::setToDefault(){

	position = default_position;
	lookAtPoint = default_lookAtPoint;
	up = default_up;

	direction = lookAtPoint - position;
	createOrth(up);

	orthoMove = vector3D( 0.f, 0.f, 1.f );

} // end set to default

void Camera::createOrth(vector3D up){

	wVec = direction * -1.0f;
	wVec.normalizeThis();

	uVec = vector3D(up.x, up.y, up.z);
	uVec = uVec % wVec;
	uVec.normalizeThis();

	vVec = wVec % uVec;
	vVec.normalizeThis();

} // end create orthographic view

void Camera::move(float u_amount, float v_amount, float w_amount) {

	vector3D uT = uVec * u_amount;
	vector3D vT = vVec * v_amount;
	vector3D wT = wVec * w_amount;

	position += uT;
	position += vT;
	position += wT;

} // end move
/*
void Camera::move(float u_amount, float v_amount) {

	vector3D uT = uVec * u_amount;
	vector3D vT = up * v_amount;

	position += uT;
	position += vT;
}
*/
void Camera::look(float vert_amount, float horz_amount) {

	vector3D tempU = vector3D(uVec.x, uVec.y, uVec.z);
	vector3D tempV = vector3D(vVec.x, vVec.y, vVec.z);
	vector3D tempW = vector3D(wVec.x, wVec.y, wVec.z);
	vector3D tempV2 = vector3D(vVec.x, vVec.y, vVec.z);
	
	// Rotate horizontally
	rotatePoint(&tempU, up, horz_amount);
	rotatePoint(&tempV, up, horz_amount);
	rotatePoint(&tempW, up, horz_amount);
	
	// Rotate vertically
	rotatePoint(&tempV2, tempU, vert_amount);
	if(tempV2.z > 0.005 && up.z > 0){	
		rotatePoint(&tempV, tempU, vert_amount);
		rotatePoint(&tempW, tempU, vert_amount);
	}
	else if(tempV2.y > 0.005 && up.y > 0){	
		rotatePoint(&tempV, tempU, vert_amount);
		rotatePoint(&tempW, tempU, vert_amount);
	}

	// Update the UVW vectors.
	uVec.x = tempU.x;
	uVec.y = tempU.y;
	uVec.z = tempU.z;
	
	vVec.x = tempV.x;
	vVec.y = tempV.y;
	vVec.z = tempV.z;
	
	wVec.x = tempW.x;
	wVec.y = tempW.y;
	wVec.z = tempW.z;
}

void Camera::rotatePoint(vector3D *pos, vector3D axis, float angle) {

	float c = cos((float)angle);
	float s = sin((float)angle);
	Matrix4 rotMat;
	
	// Setup the rotation matrix, this matrix is based off of the rotation matrix used in glRotatef.
	rotMat.a.x = axis.x * axis.x * (1 - c) + c;          	rotMat.a.y = axis.x * axis.y * (1 - c) - axis.z * s;	rotMat.a.z = axis.x * axis.z * (1 - c) + axis.y * s; 	rotMat.a.w = 0;
	rotMat.b.x = axis.y * axis.x * (1 - c) + axis.z * s; 	rotMat.b.y = axis.y * axis.y * (1 - c) + c;         	rotMat.b.z = axis.y * axis.z * (1 - c) - axis.x * s; 	rotMat.b.w = 0;
	rotMat.c.x = axis.x * axis.z * (1 - c) - axis.y * s; 	rotMat.c.y = axis.y * axis.z * (1 - c) + axis.x * s;	rotMat.c.z = axis.z * axis.z * (1 - c) + c;           	rotMat.c.w = 0;
	rotMat.d.x = 0; 					rotMat.d.y = 0; 					rotMat.d.z = 0; 					rotMat.d.w = 1;
	
	// Multiply the rotation matrix with the position vector.
	vector3D tmp;
	tmp.x = rotMat.a.x * pos->x + rotMat.a.y * pos->y + rotMat.a.z * pos->z + rotMat.a.w;
	tmp.y = rotMat.b.x * pos->x + rotMat.b.y * pos->y + rotMat.b.z * pos->z + rotMat.a.w;
	tmp.z = rotMat.c.x * pos->x + rotMat.c.y * pos->y + rotMat.c.z * pos->z + rotMat.a.w;
	
	pos->x = tmp.x;
	pos->y = tmp.y;
	pos->z = tmp.z;

}




