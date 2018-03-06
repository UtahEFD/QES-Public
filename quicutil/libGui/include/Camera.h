/* File: Camera.h
 * Author: Matthew Overby
 */

#ifndef SLUI_CAMERA_H
#define SLUI_CAMERA_H

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "Matrix4.h"
#include "OptionTracker.h"
#include "EventTracker.h"

namespace SLUI {

class Camera {
	
	public:
		/** @brief Constructs a camera from 3 vectors 
		* 
		* Creates a new camera from three vector3D's:
		* (Position, Look At Point, Up)
		*/
		Camera(vector3D pos, vector3D lap, vector3D u, OptionTracker *opt, EventTracker *evt, sf::RenderWindow *app);

		/** @brief Called in the event loop, checks to see if the event applies
		*/
		void checkEvent();

		/** @brief Sets the view based on the current options selected
		*/
		void setGLView();

		/** @brief Sets a new lookat point for the camera
		*/
		void setLookat(vector3D newLookat);

		/** @brief Sets camera to initial settings
		*/
		void setToDefault();

		/** @brief Moves the camera the amount and direction specified.
		*/
		void move(float u_amount, float v_amount, float w_amount);

		/** @brief Moves the camera the amount and direction specified.
		*/
		//void move(float u_amount, float v_amount);

		/** @brief Changes the view direction of the camera
		*/
		void look(float vert_amount, float horz_amount);

		vector3D uVec, vVec, wVec, position, lookAtPoint, direction, up;
		vector3D default_position, default_lookAtPoint, default_up;

		float farPlane;

		// Scale helps the camera move a proper distance when
		// in orthographic viewing mode.  Set it to a const number
		// depending on the x/y scale of what you render.
		float scale;

	private:
		/** @brief Rotates a vector around another at specified angle
		*/
		void rotatePoint(vector3D *pos, vector3D axis, float angle);

		/** @brief Creates and initializes the UVW vectors based on UP
		*/
		void createOrth(vector3D up);

		EventTracker *m_eventTracker;
		OptionTracker *m_optTracker;
		sf::RenderWindow *m_App;

		vector3D orthoMove;
};

}

#endif // CAMERA_H
