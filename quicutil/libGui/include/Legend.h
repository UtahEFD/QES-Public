/* File: Legend.h
 * Author: Matthew Overby
 *
 * NOTE:
 * This class isn't meant to be used for your own purposes.
 * If you look at the implementation file you'll see references
 * to options that were never initialized in Gui.cpp.
 * You should really only use it as a reference on how to
 * set up your own legend widget.
 */

#ifndef SLUI_LEGEND_H
#define SLUI_LEGEND_H

#include "Widget.h"
#include "OptionTracker.h"
#include "EventTracker.h"
#include "ColorScale.h"

namespace SLUI {

class Legend : public Widget {

	public:
		/** @brief Constructor
		*/
		Legend( EventTracker *evt, OptionTracker *opt, sf::RenderWindow* app );

		/** @brief Destructor
		*/
		~Legend();

		/** @brief Creates the gradiant background
		*/
		void createGradiant();

		void newCreateGradient();

		/** @brief Creates the labels of the legend
		*/
		void createText();

		/** @brief Draws the legend
		*/
		void draw();

		/** @brief Resizes the legend
		*/
		void resizeEvent();

		/** @brief Check if the event affects the legend
		*/
		void checkEvent();

		void setMinMaxID( std::string _minlabel, std::string _maxlabel ){
			minlabel = _minlabel;
			maxlabel = _maxlabel;
		}

		void setLegendTextID( std::string _legendtext ){
			legendtext = _legendtext;
		}

		void setPosition( int posX, int posY );

		void setColors( std::vector<sf::Color> colors );

	private:
		GLuint gradiant;
		sf::Color textColor;
		sf::String minVal, maxVal, midVal, label;

		std::string minlabel, maxlabel, legendtext;
		std::vector<sf::Color> m_colors;

		OptionTracker *m_optTracker;
		EventTracker *m_eventTracker;

		sf::Image gradientImage;
		sf::Sprite gradientSprite;
    
    bool drawable;

};

}

#endif

