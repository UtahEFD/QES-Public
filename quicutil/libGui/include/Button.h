/* File: Button.h
 * Author: Matthew Overby
 *
 * TODO:
 * SO MUCH!  Eventually most of these classes
 * should be replaced by a better button-rendering
 * library.  These are intended to be placeholders
 * until something that meshes with SLUI is found.
 */

#ifndef SLUI_BUTTON_H
#define SLUI_BUTTON_H

#include <cstdio>
#include "Widget.h"
#include "Scrollbar.h"
#include "EventTracker.h"
#include "Keys.h"
#include <sstream>

namespace SLUI {

namespace ButtonType {

	static const int Standard = 1;
	static const int Radio = 2;
	static const int Key = 3;
	static const int Value = 4;
	static const int List = 5;
	static const int Time = 6;
}

struct DropOption {

	/** @brief Constructor of an Drop button option
	*/
	DropOption(std::string newLabel, float x, float y);

	/** @brief Draw the option, indicating if it's selected
	*/
	void draw(sf::RenderWindow* m_App, bool active);

	/** @brief Highlight on mouse over
	*/
	void highlight(sf::RenderWindow* m_App);

	/** @brief Check if mouse is over on a click
	* @return boolean true if it was clicked, false otherwise
	*/
	bool mouseClicked(sf::RenderWindow *m_App);

	/** @brief Move the option to a new location
	*/
	void move(float x, float y);

	sf::String label;
	sf::Shape m_background;
	sf::Shape icon;
	sf::Shape icon_on;
	float m_posX, m_posY;
};

class Button : public Widget {

	public:
		static const int LabelSize = 18;
		static const int Width = 200;
		static const int Height = 30;

		/** @brief Called to draw the button and all of its elements
		*/
		virtual void draw() = 0;

		/** @brief Return a value associated with the button
		*  Base not implemented
		*/
		virtual float getValue();

		/** @brief Set a value associated with the button
		*  Base not implemented
		*/
		virtual void setValue(float v);

		/** @brief Set/Change the position of the button
		*/
		virtual void setPosition(float x, float y);

		/** @brief Called by checkEvent when the left mouse is clicked
		*  If mouse is over the button, toggle "active"
		*/
		virtual void onMouseClicked();

		/** @brief Called by checkEvent when the left mouse is clicked
		*  Base handles left mouse clicked only
		*/
		virtual void checkEvent();

		/** @brief If the mouse is over the button, highlight
		*/
		virtual void highlight();

		/** @brief Set/Change the label of the button
		*/
		virtual void setLabel(std::string lab);

		/** @brief Sets the size of the button background
		*  Note that label position is not changed
		*/
		virtual void setSize(float width, float height);

		/** @brief Set/Change minimum/maximum used by value buttons
		*  See the ValueButton setMinMax function for more details
		*/
		virtual void setMinMax(float min, float max);

		/** @brief Set/Change the key associated with key buttons
		*/
		virtual void setNewKey(std::string newKey);

		/** @brief Sets the droplist of a list button
		*  The old droplist is removed
		*/
		virtual void setDropList( std::vector<std::string> );

		/** @brief Set the selected drop item of a list button
		*/
		virtual void setDropSelected( std::string str );

		/** @brief Set the selected drop item of a list button to nothing
		*/
		virtual void clearSelected();

		/** @brief Return the selected drop item of a list button
		*/
		virtual std::string getDropSelected();

		/** @brief Check the type of the button.  See ButtonType
		* It's recommended you don't change this value
		*/
		int type;

		//TODO not have bool active be the value
		bool active;
		bool updated;

	protected:
		sf::String m_label;
		sf::Font m_font;
		EventTracker *m_eventTracker;

};

class StandardButton : public Button {

	public:
		/** @brief Constructor
		*/
		StandardButton(std::string lab, EventTracker *e, sf::RenderWindow* app);

		/** @brief Called to draw the button and all of its elements
		*/
		void draw();

		/** @brief Set/Change the position of the button
		*/
		void setPosition(float x, float y);

		/** @brief If the button is clicked, sets updated to true
		*/
		virtual void onMouseClicked();

	private:
};

class RadioButton : public Button {

	public:	
		/** @brief Constructor
		*/
		RadioButton(std::string lab, bool init, EventTracker *e, sf::RenderWindow* app);

		/** @brief Called to draw the button and all of its elements
		*/
		void draw();

		/** @brief Set/Change the position of the button
		*/
		void setPosition(float x, float y);

		/** @brief Returns 1 if checked/active, 0 otherwise
		*/
		float getValue();

		/** @brief Set checked/active to true (1) or false (0)
		*  If the param is not 1 or 0, nothing is changed
		*/
		void setValue(float v);

	private:
		sf::Shape icon;
		sf::Shape activeIcon;
};

class ValueButton : public Button {

	public:
		/** @brief Constructor
		*/
		ValueButton(std::string lab, float init, EventTracker *e, sf::RenderWindow* app);

		/** @brief Called to draw the button and all of its elements
		*/
		void draw();

		/** @brief Set/Change the position of the button
		*/
		void setPosition(float x, float y);

		/** @brief Check to see if the recent event applies
		*  Handles left mouse clicked and left mouse dragged
		*/
		void checkEvent();

		/** @brief Called by checkEvent on left mouse clicked
		*/
		void onMouseClicked();

		/** @brief Called by checkEvent on left mouse dragged
		*/
		void onMouseDragged();

		/** @brief Set the min and max of the value
		*  If this function is not called, the min and max will default to
		*  value > 0 : 0 to value * 2
		*  value == 0 : -1 to 1
		*  value < 0 : value * 2 to 0
		*/
		void setMinMax(float min, float max);

		/** @brief Return the current value associated with the button
		*/
		float getValue();

		/** @brief Set the value associated with the button
		*/
		void setValue(float v);

		/** @brief Changes the size of the button
		*  Width is scaled to the new size, but height is not
		*/
		void setSize(float width, float height);

		float value;

	private:
		sf::Shape icon;
		sf::Shape activeIcon;
		float maxVal, minVal;
		float sliderPosX;
		float dragSpeed;
		sf::String val;

		sf::Shape incIcon, incLab;
		sf::Shape decIcon, decLab;
};

class KeyButton : public Button {

	public:
		/** @brief Constructor
		*/
		KeyButton(std::string lab, std::string init, EventTracker *e, sf::RenderWindow* app);

		/** @brief Called to draw the button and all of its elements
		*/
		void draw();

		/** @brief Set/Change the position of the button
		*/
		void setPosition(float x, float y);

		/** @brief Check to see if the recent event applies
		*  Handles left mouse clicked and key pressed
		*/
		void checkEvent();

		/** @brief Returns the (float) sf::Key::Code of the current key
		*/
		float getValue();

		/** @brief Sets the key from the (float) sf::Key::Code
		* TODO
		*/
		void setValue(float v);

		/** @brief Sets the key by string (see Keys.h)
		*/
		void setNewKey(std::string newKey);

	private:
		sf::String key;

		/** @brief Called by checkEvent to handle a key press
		*/
		void onKeyPressed();

};

class ListButton : public Button {

	public:
		/** @brief Constructor
		*/
		ListButton(std::string lab, EventTracker *e, sf::RenderWindow* app);

		/** @brief Called to draw the button and all of its elements
		*/
		void draw();

		/** @brief Set/Change the position of the button
		*/
		void setPosition(float x, float y);

		/** @brief Check to see if the recent event applies
		*  Handles left mouse clicked
		*/
		void checkEvent();

		/** @brief Returns the number of elements in the drop list
		*/
		float getValue();

		/** @brief Sets the droplist of a list button
		*  The old droplist is removed
		*/
		void setDropList( std::vector<std::string> opts );

		/** @brief Called by checkEvent to handle a left mouse clicked
		*/
		void onMouseClicked();

		/** @brief Checks if the mouse is over and highlights elements
		*/
		void highlight();

		/** @brief Set the selected drop item of a list button
		*/
		void setDropSelected( std::string str );

		/** @brief Makes it so that no option is currently selected
		*/
		void clearSelected();

		/** @brief Return the selected drop item of a list button
		*/
		std::string getDropSelected();

		/** @brief Return the height of the button in its current state
		*/
		float getHeight();

		/** @brief Changes the size of a drop button
		*  TODO
		*/
		void setSize(float width, float height); // TODO

	private:
		/** @brief Creates a background for the drop list
		*/
		sf::Shape createListBackground(float x, float y, float height, float width);

		std::string currentListVal;
		sf::Shape icon, activeIcon;
		float listDroppedHeight;
		sf::Shape m_droppedBackground;
		std::vector< DropOption > dropOptions;

};

}

#endif

