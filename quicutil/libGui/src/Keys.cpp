/* File: Keys.cpp
 * Author: Matthew Overby
 */

#include <iostream>
#include "Keys.h"

using namespace SLUI;

Keys::Keys(){}

Keys::Keys(std::string newKeyStr){
	if(setNewKey(newKeyStr)){
		defaultKeyCode = myKeyCode;
	}
}

Keys::~Keys(){}

std::string Keys::toString(){
	return myKeyStr;
}

sf::Key::Code Keys::getKeyCode(){
	return myKeyCode;
}

bool Keys::setNewKey(std::string newKeyStr){
	sf::Key::Code newKey = findKey(newKeyStr);

	if(newKey != sf::Key::Escape){
		myKeyStr = newKeyStr;
		myKeyCode = newKey;

		return true;
	}
	else return false;
}

void Keys::toDefault(){

}

std::string Keys::keyToString(unsigned int keycode) 
{
  std::string result = "bad";
  switch (keycode) { 
    case sf::Key::A: result = "a"; break;
    case sf::Key::B: result = "b"; break;
    case sf::Key::C: result = "c"; break;
    case sf::Key::D: result = "d"; break;
    case sf::Key::E: result = "e"; break;
    case sf::Key::F: result = "f"; break;
    case sf::Key::G: result = "g"; break;
    case sf::Key::H: result = "h"; break;
    case sf::Key::I: result = "i"; break;
    case sf::Key::J: result = "j"; break;
    case sf::Key::K: result = "k"; break;
    case sf::Key::L: result = "l"; break;
    case sf::Key::M: result = "m"; break;
    case sf::Key::N: result = "n"; break;
    case sf::Key::O: result = "o"; break;
    case sf::Key::P: result = "p"; break;
    case sf::Key::Q: result = "q"; break;
    case sf::Key::R: result = "r"; break;
    case sf::Key::S: result = "s"; break;
    case sf::Key::T: result = "t"; break;
    case sf::Key::U: result = "u"; break;
    case sf::Key::V: result = "v"; break;
    case sf::Key::W: result = "w"; break;
    case sf::Key::X: result = "x"; break;
    case sf::Key::Y: result = "y"; break;
    case sf::Key::Z: result = "z"; break;
    case sf::Key::Num0: result = "0"; break;
    case sf::Key::Num1: result = "1"; break;
    case sf::Key::Num2: result = "2"; break;
    case sf::Key::Num3: result = "3"; break;
    case sf::Key::Num4: result = "4"; break;
    case sf::Key::Num5: result = "5"; break;
    case sf::Key::Num6: result = "6"; break;
    case sf::Key::Num7: result = "7"; break;
    case sf::Key::Num8: result = "8"; break;
    case sf::Key::Num9: result = "9"; break;
    case sf::Key::LShift: result = "lshift"; break; 
    case sf::Key::LAlt: result = "lalt"; break; 
    case sf::Key::RShift: result = "rshift"; break; 
    case sf::Key::RAlt: result = "ralt"; break; 
    case sf::Key::LBracket: result = "["; break; 
    case sf::Key::RBracket: result = "]"; break; 
    case sf::Key::SemiColon: result = ";"; break; 
    case sf::Key::Comma: result = ","; break; 
    case sf::Key::Period: result = "."; break; 
    case sf::Key::Quote: result = "\'"; break; 
    case sf::Key::Slash: result = "/"; break; 
    case sf::Key::Tilde: result = "~"; break; 
    case sf::Key::Equal: result = "="; break; 
    case sf::Key::Dash: result = "-"; break;
    case sf::Key::Space: result = "space"; break; 
    case sf::Key::Return: result = "enter"; break; 
    case sf::Key::Back: result = "backspace"; break; 
    case sf::Key::PageUp: result = "pageup"; break; 
    case sf::Key::PageDown: result = "pagedown"; break; 
    case sf::Key::End: result = "end"; break; 
    case sf::Key::Home: result = "home"; break; 
    case sf::Key::Insert: result = "insert"; break; 
    case sf::Key::Delete: result = "delete"; break; 
    case sf::Key::Add: result = "+"; break; 
    case sf::Key::Subtract: result = "-"; break; 
    case sf::Key::Multiply: result = "*"; break; 
    case sf::Key::Divide: result = "/"; break; 
    case sf::Key::Left: result = "left"; break; 
    case sf::Key::Right: result = "right"; break; 
    case sf::Key::Up: result = "up"; break; 
    case sf::Key::Down: result = "down"; break; 
    case sf::Key::Numpad0: result = "num0"; break; 
    case sf::Key::Numpad1: result = "num1"; break; 
    case sf::Key::Numpad2: result = "num2"; break; 
    case sf::Key::Numpad3: result = "num3"; break; 
    case sf::Key::Numpad4: result = "num4"; break; 
    case sf::Key::Numpad5: result = "num5"; break; 
    case sf::Key::Numpad6: result = "num6"; break; 
    case sf::Key::Numpad7: result = "num7"; break; 
    case sf::Key::Numpad8: result = "num8"; break; 
    case sf::Key::Numpad9: result = "num9"; break; 
    case sf::Key::F1: result = "f1"; break; 
    case sf::Key::F2: result = "f2"; break; 
    case sf::Key::F3: result = "f3"; break; 
    case sf::Key::F4: result = "f4"; break; 
    case sf::Key::F5: result = "f5"; break; 
    case sf::Key::F6: result = "f6"; break; 
    case sf::Key::F7: result = "f7"; break; 
    case sf::Key::F8: result = "f8"; break; 
    case sf::Key::F9: result = "f9"; break; 
    case sf::Key::F10: result = "f10"; break; 
    case sf::Key::F11: result = "f11"; break; 
    case sf::Key::F12: result = "f12"; break; 
    case sf::Key::F13: result = "f13"; break; 
    case sf::Key::F14: result = "f14"; break; 
    case sf::Key::F15: result = "f15"; break;
  }
  return result;
}

sf::Key::Code Keys::findKey(std::string newKey)
{
  sf::Key::Code result = sf::Key::Escape;
  if(newKey.length() < 2){
    char newChar = newKey.at(0);
    switch(newChar){
      case 0:
	break;
      case 'a':
	result = sf::Key::A;
	break;
      case 'b':
	result = sf::Key::B;
	break;
      case 'c':
	result = sf::Key::C;
	break;
      case 'd':
	result = sf::Key::D;
	break;
      case 'e':
	result = sf::Key::E;
	break;
      case 'f':
	result = sf::Key::F;
	break;
      case 'g':
	result = sf::Key::G;
	break;
      case 'h':
	result = sf::Key::H;
	break;
      case 'i':
	result = sf::Key::I;
	break;
      case 'j':
	result = sf::Key::J;
	break;
      case 'k':
	result = sf::Key::K;
	break;
      case 'l':
	result = sf::Key::L;
	break;
      case 'm':
	result = sf::Key::M;
	break;
      case 'n':
	result = sf::Key::N;
	break;
      case 'o':
	result = sf::Key::O;
	break;
      case 'p':
	result = sf::Key::P;
	break;
      case 'q':
	result = sf::Key::Q;
	break;
      case 'r':
	result = sf::Key::R;
	break;
      case 's':
	result = sf::Key::S;
	break;
      case 't':
	result = sf::Key::T;
	break;
      case 'u':
	result = sf::Key::U;
	break;
      case 'v':
	result = sf::Key::V;
	break;
      case 'w':
	result = sf::Key::W;
	break;
      case 'x':
	result = sf::Key::X;
	break;
      case 'y':
	result = sf::Key::Y;
	break;
      case 'z':
	result = sf::Key::Z;
	break;
      case '0':
	result = sf::Key::Num0;
	break;
      case '1':
	result = sf::Key::Num1;
	break;
      case '2':
	result = sf::Key::Num2;
	break;
      case '3':
	result = sf::Key::Num3;
	break;
      case '4':
	result = sf::Key::Num4;
	break;
      case '5':
	result = sf::Key::Num5;
	break;
      case '6':
	result = sf::Key::Num6;
	break;
      case '7':
	result = sf::Key::Num7;
	break;
      case '8':
	result = sf::Key::Num8;
	break;
      case '9':
	result = sf::Key::Num9;
	break;
      case '[':
	result = sf::Key::LBracket;
	break;
      case ']':
	result = sf::Key::RBracket;
	break;
      case ';':
	result = sf::Key::SemiColon;
	break;
      case '.':
	result = sf::Key::Period;
	break;
      case ',':
	result = sf::Key::Comma;
	break;
      case '/':
	result = sf::Key::Slash;
	break;
      case '~':
	result = sf::Key::Tilde;
	break;
      case '`':
	result = sf::Key::Quote;
	break;
      case '=':
	result = sf::Key::Equal;
	break;
      case '-':
	result = sf::Key::Dash;
	break;
      case '+':
	result = sf::Key::Add;
	break;
      case '*':
	result = sf::Key::Multiply;
	break;
    }
  }
  else{

    if(newKey.compare("lshift")==0){
      result = sf::Key::LShift;
    }
    else if(newKey.compare("lalt")==0){
      result = sf::Key::LAlt;
    }
    else if(newKey.compare("rshift")==0){
      result = sf::Key::RShift;
    }
    else if(newKey.compare("ralt")==0){
      result = sf::Key::RAlt;
    }
    else if(newKey.compare("space")==0){
      result = sf::Key::Space;
    }
    else if(newKey.compare("enter")==0){
      result = sf::Key::Return;
    }
    else if(newKey.compare("backspace")==0){
      result = sf::Key::Back;
    }
    else if(newKey.compare("pageup")==0){
      result = sf::Key::PageUp;
    }
    else if(newKey.compare("pagedown")==0){
      result = sf::Key::PageDown;
    }
    else if(newKey.compare("end")==0){
      result = sf::Key::End;
    }
    else if(newKey.compare("home")==0){
      result = sf::Key::Home;
    }
    else if(newKey.compare("insert")==0){
      result = sf::Key::Insert;
    }
    else if(newKey.compare("delete")==0){
      result = sf::Key::Delete;
    }
    else if(newKey.compare("num0")==0){
      result = sf::Key::Delete;
    }
    else if(newKey.compare("num0")==0){
      result = sf::Key::Numpad0;
    }
    else if(newKey.compare("num1")==0){
      result = sf::Key::Numpad1;
    }
    else if(newKey.compare("num2")==0){
      result = sf::Key::Numpad2;
    }
    else if(newKey.compare("num3")==0){
      result = sf::Key::Numpad3;
    }
    else if(newKey.compare("num4")==0){
      result = sf::Key::Numpad4;
    }
    else if(newKey.compare("num5")==0){
      result = sf::Key::Numpad5;
    }
    else if(newKey.compare("num6")==0){
      result = sf::Key::Numpad6;
    }
    else if(newKey.compare("num7")==0){
      result = sf::Key::Numpad7;
    }
    else if(newKey.compare("num8")==0){
      result = sf::Key::Numpad8;
    }
    else if(newKey.compare("num9")==0){
      result = sf::Key::Numpad9;
    }
    else if(newKey.compare("up")==0){
      result = sf::Key::Up;
    }
    else if(newKey.compare("down")==0){
      result = sf::Key::Down;
    }
    else if(newKey.compare("left")==0){
      result = sf::Key::Left;
    }
    else if(newKey.compare("right")==0){
      result = sf::Key::Right;
    }
    else if(newKey.compare("f1")==0){
      result = sf::Key::F1;
    }
    else if(newKey.compare("f2")==0){
      result = sf::Key::F2;
    }
    else if(newKey.compare("f3")==0){
      result = sf::Key::F3;
    }
    else if(newKey.compare("f4")==0){
      result = sf::Key::F4;
    }
    else if(newKey.compare("f5")==0){
      result = sf::Key::F5;
    }
    else if(newKey.compare("f6")==0){
      result = sf::Key::F6;
    }
    else if(newKey.compare("f7")==0){
      result = sf::Key::F7;
    }
    else if(newKey.compare("f8")==0){
      result = sf::Key::F8;
    }
    else if(newKey.compare("f9")==0){
      result = sf::Key::F9;
    }
    else if(newKey.compare("f10")==0){
      result = sf::Key::F10;
    }
    else if(newKey.compare("f11")==0){
      result = sf::Key::F11;
    }
    else if(newKey.compare("f12")==0){
      result = sf::Key::F12;
    }
    else if(newKey.compare("f13")==0){
      result = sf::Key::F13;
    }
    else if(newKey.compare("f14")==0){
      result = sf::Key::F14;
    }
    else if(newKey.compare("f15")==0){
      result = sf::Key::F15;
    }
  }
  return result;
}


