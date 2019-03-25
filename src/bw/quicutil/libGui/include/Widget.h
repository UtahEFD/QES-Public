/* File: Widget.h
 * Author: Matthew Overby
 *
 * TODO:
 * A better scheme for margin/padding that
 * mimics CSS.
 */

#ifndef SLUI_WIDGET_H
#define SLUI_WIDGET_H

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <sstream>
#include <iostream>

namespace SLUI {

namespace TextAlign {

	static const int Center = 1;
	static const int Left = 2;
	static const int Right = 3;
	static const int Top = 4;
	static const int Bottom = 5;
}

class Widget {

	public:
		/** @brief Default Constructor
		*/
		Widget();

		/** @brief Default Destructor
		*/
		virtual ~Widget();

		/** @brief Changes window colors with (border, background, highlight, text)
		 */
		virtual void changeColors(sf::Color border, sf::Color bg, sf::Color highlight, sf::Color text);

		/** @brief Draws the widget (pure virtual method)
		*/
		virtual void draw() = 0;

		/** @brief Find out if the mouse is over the widget (based on m_pos and m_width/height)
		* @return Boolean value true if it is, false otherwise
		*/
		virtual bool isMouseOver();

		/** @brief Find out if the mouse is over a particular posX, posY, width and height
		* @return Boolean value true if it is, false otherwise
		*/
		virtual bool isMouseOver(int posX, int posY, int width, int height);

		/** @brief return the position of the widget as a 2D vector
		* @return 2D SFML vector
		*/
		virtual sf::Vector2f getPosition() const { return sf::Vector2f(m_posX, m_posY); }

		/** @brief Get the width of the widget
		* @return float value representing m_width
		*/
		virtual float getWidth() const { return m_width; }

		/** @brief Get the height of the widget
		* @return float value representing m_height
		*/
		virtual float getHeight() const { return m_height; }

		/** @brief Set whether or not the widget is highlighted
		*/
		void setHighlight(bool val);

		virtual sf::Vector2f getPosition(){ 
			sf::Vector2f pos = sf::Vector2f( m_posX, m_posY );
			return pos;
		}

		virtual void setPadding( int padT, int padR, int padB, int padL ){ 
			m_paddingL = padL; 
			m_paddingR = padR;
			m_paddingT = padT;
			m_paddingB = padB;
		}

		/** @brief Sets the position of the widget
		*/
		//void setPosition( int posX, int posY );

	protected:
		/** @brief creates a background for the widget
		*
		*  Creates an SFML rectangle for the background of the widget
		*  with (position X, position Y, height, width)
		*/
		sf::Shape createBackground(float, float, float, float);

		sf::Shape m_background;
		sf::RenderWindow* m_App;
		sf::Color m_bgColor;
		sf::Color m_borderColor;
		sf::Color m_highlightColor;
		sf::Color m_textColor;

		float m_height, m_width, m_posX, m_posY;
		float m_paddingL, m_paddingR, m_paddingT, m_paddingB;
};

}

#endif

