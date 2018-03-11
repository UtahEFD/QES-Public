/* File: LoadScreen.h
 * Author: Matthew Overby
 *
 * TODO:
 * Push this whole class into the WindowManager class so the
 * user doesn't have to set it up.  It will be much easier
 * to call a function like "drawLoadscreen( ... )".
 */

#ifndef SLUI_LOADSCREEN_H
#define SLUI_LOADSCREEN_H

#include "Widget.h"

namespace SLUI {

class LoadScreen : public Widget{

	public:
		/** @brief Constructor
		*/
		LoadScreen(sf::RenderWindow* app);

		/** @brief Draws the load screen
		*/
		void draw();

		/** @brief Resizes the load screen on a resize event
		*/
		void resizeEvent();

	private:
		sf::String loading;

};

}

#endif

