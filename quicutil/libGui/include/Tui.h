/* File: Tui.h
 * Author: Matthew Overby
 *
 * TODO:
 * This is a very rough first draft of a textual user interface.
 * An example of how to use this class is in samples/SampleTui.
 */

#ifndef SLUI_TUI_H
#define SLUI_TUI_H

#include <sstream>
#include <curses.h>
#include <signal.h>
#include <iostream>
#include <vector>
#include <map>
#include "OptionTracker.h"

namespace SLUI {

class ChoiceTracker {

	public:
		ChoiceTracker();
		void addChoice( std::string label, int id );
		void select( std::string label );
		int getSelected();

	private:
		std::map< std::string, int > choices;
		int lastSelected;
};

struct Setting {

	static const int Choice = 1;
	static const int Boolean = 2;
	static const int Value = 3;

	Setting();
	Setting( std::string lab, int t );
	std::string label;
	int type;
	bool active;
};

class Menu {

	public:
		virtual void draw() = 0;
		virtual void open( int ) = 0;
		virtual void close() = 0;
		virtual void checkEvent() = 0;
		virtual int getWidth(){ return m_width; }
		virtual void addChoices( std::vector< std::string > menu_choices ){}
		virtual void addSettings( std::vector< Setting > menu_settings ){}
		bool isOpen;

	protected:
		ChoiceTracker *m_choiceTracker;
		int selected;
		WINDOW *window;
		int m_width;
		int m_height;
		int m_posX;
		int m_posY;
		int id;
		std::string label;
};

class WindowTracker {

	public:
		WindowTracker();
		~WindowTracker();
		void draw();
		void checkEvent();
		void addWindow( std::string label, Menu* newMenu );
		void open( std::string label );
		void closeTop();
		void clearMsgWindow();
		std::streambuf *coutStreamBuff;
		std::stringstream *coutBuffer;

	private:
		std::vector< Menu* > windowStack;
		std::map< std::string, Menu* > windows;
		int m_width;
		int m_height;
		WINDOW *msgWindow;
		int windowBottom;
};

class TextMenu : public Menu {

	public:
		TextMenu( std::string lbl, int _id, int h, int w, ChoiceTracker *chT, std::vector< std::string > menu_choices );
		~TextMenu();
		void open( int posY );
		void close();
		void draw();
		void checkEvent();

	private:
		std::vector< std::string > menuChoices;
};

class SettingsMenu : public Menu {

	public:
		SettingsMenu( std::string lbl, int _id, int h, int w, ChoiceTracker *chT, 
			OptionTracker *opT, std::vector< Setting > menu_settings );
		~SettingsMenu();
		void open( int posY );
		void close();
		void draw();
		void checkEvent();

	private:
		OptionTracker *m_optTracker;
		std::vector< Setting > menuSettings;
};

class Tui {

	public:
		/** @brief Constructor
		*/
		Tui();

		/** @brief Destructor
		*/
		~Tui();

		/** @brief Initializes the screen and opens the main menu
		*/
		void startTui();

		/** @brief Clears the screen and quits ncurses
		*/
		void closeTui();

	protected:
		ChoiceTracker *m_choiceTracker;
		OptionTracker *m_optTracker;
		WindowTracker *m_winTracker;
};

}

#endif
