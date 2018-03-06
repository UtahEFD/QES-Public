/* File: Window.cpp
 * Author: Matthew Overby
 * 
 * TODO:
 * "Not implemented" error messages
 */

#include "Window.h"

using namespace SLUI;

void WindowController::open(int window){

	std::vector<int>::iterator it = std::find(m_openStack.begin(), m_openStack.end(), window);
	if(it == m_openStack.end()) m_openStack.push_back(window);
}

void WindowController::close(int window){

	std::vector<int>::iterator it = std::find(m_openStack.begin(), m_openStack.end(), window);
	if(it != m_openStack.end()) m_openStack.erase(it);
}

int Window::getId(){
	return id;
}

void Window::close(){

	if( m_winController ) m_winController->close( id );
}

void Window::open(){

	if( m_winController ) m_winController->open( id );
}

void Window::highlight(){}

void Window::update(){}

void Window::resizeEvent(){}

void Window::loadText( std::string file ){}

void Window::setCellFuncLabel( std::string label ){}
