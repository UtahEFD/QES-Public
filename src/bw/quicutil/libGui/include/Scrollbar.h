/* File: Scrollbar.h
 * Author: Matthew Overby
 *
 * TODO:
 * Make scrollbars draggable.  The functionality is
 * available in EventTracker, with an example in Button::ValueButton.
 * Also, needs revision on how it determines slider length.
 * Scrollbar should be determined based off pixel height, not
 * number of lines.
 */

#ifndef SLUI_SCROLLBAR_H
#define SLUI_SCROLLBAR_H

#include "Widget.h"
#include "EventTracker.h"

namespace SLUI {

class Scrollbar : public Widget {

	public:
		/** @brief Constructor
		*/
		Scrollbar();

		/** @brief Constructor
		*
		* Creates a new scroll bar with positions x and y, its height, and a
		* pointer to the render window.
		*/
		Scrollbar(int, int, int, sf::RenderWindow*);

		/** @brief Destructor
		*/
		~Scrollbar();

		/** @brief Creates a slider
		*
		* Creates a slider based on the total number of lines and visible number
		* of lines.
		*/
		void makeSlider(int newLinesTotal, int newLinesVisible);

		/** @brief Highlights up and down arrows if the mouse if over them
		*/
		void highlight(const sf::Input *);

		/** @brief Checks to see if the mouse is over the up arrow
		* @return true if it is, false otherwise
		*/
		bool isMouseOverUp(const sf::Input *);

		/** @brief Checks to see if the mouse is over the down arrow
		* @return true if it is, false otherwise
		*/
		bool isMouseOverDown(const sf::Input *);

		/** @brief Checks to see if the mouse is over the bar
		* @return true if it is, false otherwise
		*/
		bool isMouseOverBar();

		/** @brief Scrolls the text up or down based on the parameter
		* 
		* If you want to scroll up on the text, set the parameter to true,
		* to scroll down, false.
		*/
		void scroll(bool up);

		/** @brief Moves the slider to the top position
		*/
		void moveSliderToTop();

		/** @brief Resize the scrollbar with a new xpos, ypos, and height
		*
		* Scrollbar's resize event is different than other widget classes.
		* Instead of giving it the window's new height and width, it gets
		* a new position and height.
		*/
		void resizeEvent(int, int, int);

		/** @brief Draws all scroll bar elements
		*/
		void draw();

		void checkEvent();

	private:
		sf::Shape slider;
		sf::Shape upArrow;
		sf::Shape downArrow;
		sf::Shape upBg;
		sf::Shape downBg;

		float linesVisible;
		float linesTotal;
		float sliderPosY, sliderHeight;
};

}

#endif
