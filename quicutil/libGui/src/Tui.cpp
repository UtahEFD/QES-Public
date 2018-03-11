/* File: Tui.cpp
 * Author: Matthew Overby
 */

#include "Tui.h"

using namespace SLUI;


/*********************
*	CHOICE TRACKER
*********************/


ChoiceTracker::ChoiceTracker(){}


void ChoiceTracker::addChoice( std::string label, int id ){

	std::map< std::string, int >::iterator it = choices.find( label );
	if(it == choices.end()){
		choices.insert( std::pair< std::string, int >( label, id ) );
	}
}

void ChoiceTracker::select( std::string label ){

	std::map< std::string, int >::iterator it = choices.find( label );
	if(it != choices.end()){
		lastSelected = it->second;
	}
}

int ChoiceTracker::getSelected(){

	int result = -1;
	if( lastSelected >= 0 ){
		result = lastSelected;
	}
	lastSelected = -1;

	return result;
}


/*********************
*	WINDOW TRACKER
*********************/


WindowTracker::WindowTracker(){

	m_height = 0;
	m_width = 0;
	coutStreamBuff = 0;
}

WindowTracker::~WindowTracker(){

	std::map< std::string, Menu* >::iterator it;
	for ( it = windows.begin(); it != windows.end(); it++ ) {
		Menu *delPtr = it->second;
		if( delPtr ){
			delete delPtr;
			windows.erase(it);
		}
	}
	delwin( msgWindow );
}

void WindowTracker::draw(){

	int row, col;
	getmaxyx( stdscr, row, col );
	if( m_height != row || m_width != col ){
		m_height = row;
		m_width = col;
		clear();
		windowBottom = row/2;
		delwin( msgWindow );
		msgWindow = newwin( row/2-2, col-4, windowBottom, 2 );
	}

	for( int i=0; i < windowStack.size(); i++ ){
		windowStack[i]->draw();
	}
	box(msgWindow, 0, 0);
	wrefresh(msgWindow);

	// Write out the cout stream buffer
	if( coutStreamBuff && coutBuffer ){

		std::stringstream out;
		out << coutBuffer->str();
		int index = 0;
		std::string line;
		while( std::getline(out, line) ){
			index++;
			if( index < row ){
				mvwprintw(msgWindow, index, 1, "%s", line.c_str() );
			}
		}

		coutBuffer->str( std::string() );
		coutBuffer->clear();

	} // end streambuffer exists
}

void WindowTracker::checkEvent(){

	if( windowStack.size() > 0 ){
		windowStack.back()->checkEvent();
	}
}

void WindowTracker::addWindow( std::string label, Menu* newMenu ){

	windows.insert( std::pair< std::string, Menu* >( label, newMenu ) );
}

void WindowTracker::open( std::string label ){

	std::map< std::string, Menu* >::iterator window = windows.find( label );
	if( window != windows.end() ){

		int openPosY = 2;
		std::vector< Menu* >::iterator it;

		for( it = windowStack.begin(); it < windowStack.end(); it++ ){
			openPosY += (*it)->getWidth()+2;
		}

		window->second->open( openPosY );
		windowStack.push_back( window->second );
	}
}

void WindowTracker::closeTop(){

	if( windowStack.size() > 0 ){
		windowStack.back()->close();
		windowStack.pop_back();
	}
}

void WindowTracker::clearMsgWindow(){

	// Clear the current message window
	int msgRow, msgCol;
	getmaxyx( msgWindow, msgRow, msgCol );
	for( int y=0; y<msgRow; y++ ){
		for( int x=0; x<msgCol; x++ ){
			mvwprintw(msgWindow, y, x, "%s", " " );
		}
	}
}


/*********************
*	TUI
*********************/


Tui::Tui(){

	m_choiceTracker = new ChoiceTracker();
	m_optTracker = new OptionTracker();
	m_winTracker = new WindowTracker();
}

Tui::~Tui(){

	delete m_optTracker;
	delete m_choiceTracker;
	delete m_winTracker;
}

void Tui::startTui(){

	initscr(); // initialize screen
	clear();
	noecho();
	cbreak();
	curs_set( 0 ); // hide cursor
}

void Tui::closeTui(){

	clrtoeol();
	refresh();
	endwin();
}


/*********************
*	TEXT MENU
*********************/


TextMenu::TextMenu( std::string lbl, int _id, int h, int w, ChoiceTracker *chT, std::vector< std::string > menu_choices ){

	m_height = h;
	m_width = w;
	id = _id;
	label = lbl;
	window = 0;
	selected = 0;
	m_choiceTracker = chT;
	isOpen = false;

	menuChoices = menu_choices;
	m_height = 4+menuChoices.size();
}

TextMenu::~TextMenu(){

	if( window ) delwin( window );
}


void TextMenu::open( int posY ){

	isOpen = true;
	int m_posX = 6;
	int m_posY = posY;
	window = newwin( m_height, m_width, m_posX, m_posY );
	keypad(window, TRUE);
}

void TextMenu::close(){

	isOpen = false;
	delwin( window );
	window = 0;
}

void TextMenu::draw(){

	if( isOpen ){
		int x, y, i;	

		x = 2;
		y = 2;
		box(window, 0, 0);
		for(i = 0; i < menuChoices.size(); ++i){

			if(selected == i){ // Highlight the present choice

				wattron(window, A_REVERSE); 
				mvwprintw(window, y, x, "%s", menuChoices[i].c_str() );
				wattroff(window, A_REVERSE);
			}
			else
				mvwprintw(window, y, x, "%s", menuChoices[i].c_str() );
			++y;
		}
		wrefresh(window);
	}
}

void TextMenu::checkEvent(){

	int c = wgetch(window);

	switch( c ){

		case KEY_UP:{
			if(selected == 0)
				selected = menuChoices.size()-1;
			else
				--selected;
		} break;

		case KEY_DOWN:{
			if(selected == menuChoices.size()-1)
				selected = 0;
			else 
				++selected;
		} break;

		case 10:{ // enter
			std::string selectedStr = menuChoices[selected];
			m_choiceTracker->select( selectedStr );
		} break;

		default:{
			refresh();
		} break;

	}
}


/*********************
*	SETTINGS MENU
*********************/


Setting::Setting(){

	label = "";
	active = false;
}

Setting::Setting( std::string lab, int t ){

	label = lab;
	type = t;
	active = false;
}

SettingsMenu::SettingsMenu( std::string lbl, int _id, int h, int w, ChoiceTracker *chT, 
	OptionTracker *opT, std::vector< Setting > menu_settings ){

	m_height = h;
	m_width = w;
	id = _id;
	label = lbl;
	window = 0;
	selected = 0;
	m_choiceTracker = chT;
	m_optTracker = opT;
	isOpen = false;

	menuSettings = menu_settings;
	m_height = 4+menuSettings.size();
}

SettingsMenu::~SettingsMenu(){

	if( window ) delwin( window );
}

void SettingsMenu::open( int posY ){

	isOpen = true;
	int m_posX = 6;
	int m_posY = posY;
	window = newwin( m_height, m_width, m_posX, m_posY );
	keypad(window, TRUE);
}

void SettingsMenu::close(){

	isOpen = false;
	delwin( window );
	window = 0;
}

void SettingsMenu::draw(){

	if( isOpen ){
		int x, y, i;	

		x = 2;
		y = 2;
		box(window, 0, 0);
		wrefresh(window);
		for( i = 0; i < menuSettings.size(); i++ ){

			if(selected == i){ // Highlight the present choice
				if( !menuSettings[i].active ){

					wattron(window, A_REVERSE ); 
					mvwprintw(window, y, x, "%s", menuSettings[i].label.c_str() );
					wattroff(window, A_REVERSE );

					std::string value = m_optTracker->getString( menuSettings[i].label );
					mvwprintw(window, y, x+16, "%s   ", value.c_str() );
				}
				else{

					mvwprintw(window, y, x, "%s", menuSettings[i].label.c_str() );

					wattron(window, A_REVERSE );
					std::string value = m_optTracker->getString( menuSettings[i].label );
					mvwprintw(window, y, x+16, "%s  ", value.c_str() );
					wattroff(window, A_REVERSE );
				}
			}
			else{
				mvwprintw(window, y, x, "%s", menuSettings[i].label.c_str() );
				std::string value = m_optTracker->getString( menuSettings[i].label );
				mvwprintw(window, y, x+16, "%s   ", value.c_str() );
			}
			y++;
		}
		wrefresh(window);
	}
}

void SettingsMenu::checkEvent(){

	int c = wgetch(window);
	Setting curr = menuSettings[selected];

	switch( c ){

		case KEY_UP:{
			if( curr.type == curr.Value && curr.active ){
				float tempVal = m_optTracker->getValue( curr.label ) + 1.f;
				m_optTracker->setValue( curr.label, tempVal );
			}
			else if( curr.type == curr.Boolean && curr.active ){
				m_optTracker->toggle( curr.label );
			}
			else{
				if(selected == 0)
					selected = menuSettings.size()-1;
				else
					--selected;
			}
		} break;

		case KEY_DOWN:{
			if( curr.type != curr.Choice && curr.active ){
				float tempVal = m_optTracker->getValue( curr.label ) - 1.f;
				m_optTracker->setValue( curr.label, tempVal );
			}
			else if( curr.type == curr.Boolean && curr.active ){
				m_optTracker->toggle( curr.label );
			}
			else {
				if(selected == menuSettings.size()-1)
					selected = 0;
				else 
					++selected;
			}
		} break;

		case 10:{ // enter
			if( curr.type != curr.Choice && curr.active ){
				menuSettings[selected].active = false;
			}
			else if( curr.type != curr.Choice && !curr.active ){
				menuSettings[selected].active = true;
			}
			else{
				std::string selectedStr = curr.label;
				m_choiceTracker->select( selectedStr );
			}
			refresh();
		} break;

		default:{
			refresh();
		} break;

	}
}



