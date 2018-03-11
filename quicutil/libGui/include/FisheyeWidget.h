/* File: FisheyeWidget.h
 * Author: Matthew Overby
 *
 * NOTE:
 * This class isn't meant to be used for your own purposes.
 * If you look at the implementation file you'll see references
 * to options that were never initialized in Gui.cpp.
 * You should really only use it as a reference on how to
 * set up your own FisheyeWidget widget.
 */

#ifndef SLUI_FISHEYEWIDGET_H
#define SLUI_FISHEYEWIDGET_H

#include "Widget.h"
#include "OptionTracker.h"
#include "EventTracker.h"
#include "ColorScale.h"
#include "vector3D.h"

namespace SLUI {

class FisheyeWidget : public Widget {

	public:
		/** @brief Constructor
		*/
		FisheyeWidget( EventTracker *evt, OptionTracker *opt, sf::RenderWindow* app );

		/** @brief Destructor
		*/
		~FisheyeWidget();

		/** @brief Draws the FisheyeWidget
		*/
		void draw();

		/** @brief Resizes the FisheyeWidget
		*/
		void resizeEvent();

		/** @brief Check if the event affects the FisheyeWidget
		*/
		void checkEvent();

		void setPosition( int posX, int posY );

		void setData( std::map< std::pair<int,int>, vector3D > newdata );

	private:
		OptionTracker *m_optTracker;
		EventTracker *m_eventTracker;
		sf::Sprite m_sprite;
		sf::Image m_image;
		sf::Color *colors;

};

	typedef std::map< std::pair<int,int>, vector3D > PixelMap;

}

#endif

