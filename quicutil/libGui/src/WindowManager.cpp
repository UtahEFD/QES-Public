/* File: WindowManager.cpp
 * Author: Matthew Overby
 */

#include "WindowManager.h"

using namespace SLUI;

WindowManager::WindowManager(WindowController *wnC, EventTracker *evt, OptionTracker *opT, sf::RenderWindow *app){

	m_optTracker = opT;
	m_winController = wnC;
	m_eventTracker = evt;
	m_App = app;
	m_console = new Console(CONSOLE, m_winController, m_optTracker, m_eventTracker, m_App);
	addWindow( m_console );
}

WindowManager::~WindowManager(){

	std::map< int, Window* >::iterator it;
	for ( it=m_winTable.begin(); it != m_winTable.end(); it++ ) {

		Window *delPtr = it->second;
		if( delPtr ){
			delete delPtr;
			m_winTable.erase(it);
		}
	}
}

void WindowManager::addWindow(Window* newWin){

	if( newWin ){
		int id = newWin->getId();
		if( id != 0 ){
			m_winTable[ id ] = newWin;
		}
	}
}

void WindowManager::open(int window){

	std::map<int, Window*>::iterator winIter = m_winTable.find( window );
	if(winIter != m_winTable.end()){ winIter->second->open(); }
}

void WindowManager::close(int window){

	std::map<int, Window*>::iterator winIter = m_winTable.find( window );
	if(winIter != m_winTable.end()){ winIter->second->close(); }
}

bool WindowManager::isOpen(){

	if(m_winController->m_openStack.size() > 0) return true;
	else return false;
}

bool WindowManager::isTop(int window){

	std::vector<int>::iterator it = std::find(m_winController->m_openStack.begin(), 
		m_winController->m_openStack.end(), window);

	if (it != m_winController->m_openStack.end()) return *it == m_winController->m_openStack.back();
	return false;
}

bool WindowManager::isOpen(int window){

	return ( std::find(m_winController->m_openStack.begin(), m_winController->m_openStack.end(), window)
		!= m_winController->m_openStack.end() );
}

int WindowManager::getTop(){

	if(m_winController->m_openStack.size() == 0) return 0;
	else return m_winController->m_openStack.back();
}

void WindowManager::drawWindows(){

	m_console->draw();
	std::map<int, Window*>::iterator iter;
	for(int i=0; i < m_winController->m_openStack.size(); i++){

		iter = m_winTable.find( m_winController->m_openStack.at(i) );
		if(iter != m_winTable.end()){
			iter->second->draw();
		}
	}
}

void WindowManager::checkEvent(){

	if(m_eventTracker->messageTime > 0){
		m_console->showMessage( m_eventTracker->message, m_eventTracker->messageTime );
		m_eventTracker->messageTime = 0;
	}

	int eventType = m_eventTracker->eventType;
	if(!m_optTracker->getActive("hidevisuals")){

		std::map<int, Window*>::iterator iter;

		if(eventType == EventType::W_Resize){

			for(iter=m_winTable.begin(); iter!=m_winTable.end(); iter++){
				iter->second->resizeEvent();
			}

		}
		else if( m_winController->m_openStack.size() > 0 ) {

			iter = m_winTable.find( m_winController->m_openStack.back() );
			if(iter != m_winTable.end()){
				iter->second->checkEvent();
				iter->second->highlight();
			}

		}
	}
}

void WindowManager::update(){

	std::map<int, Window*>::iterator iter;

	for(iter=m_winTable.begin(); iter!=m_winTable.end(); iter++){
		iter->second->update();
	}
}

void WindowManager::update(int id){

	std::map<int, Window*>::iterator it = m_winTable.find(id);
	if (it != m_winTable.end()) it->second->update();
}

Window* WindowManager::getWindow(int id){

	std::map<int, Window*>::iterator it = m_winTable.find(id);
	if (it != m_winTable.end()) return it->second;
	else return ( 0 );
}

void WindowManager::loadText(int id, std::string file){

	FileFinder fileFinder( ".." );
	std::string path = fileFinder.findRecursive( file );
	std::map<int, Window*>::iterator it = m_winTable.find(id);
	if (it != m_winTable.end()){
		it->second->loadText( path );
	} // end window found
}



