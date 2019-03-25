
SIVELAB User Interface (SLUI)

Matthew Overby, over0219@d.umn.edu
Department of Computer Science
University of Minnesota Duluth


1	== OVERVIEW ==

SLUI is a user interface library that was originally created for the project QUIC Energy at the University of Minnesota Duluth.
Its purpose is to allow general program functionality and menuing, handle controls and input, allow user-customizable configurations,
and provide simple HUD-like widgets.  A brief list of features is shown in section 4 (FEATURES).

Using SFML 1.6, OpenGL, and ncurses, SLUI will generate a default menu, with many of the basic default options and windows.  It was designed
to be extendable for other uses and provide instructions on how to utilize its features.  These instructions can be found in the 
"samples" directory.

Since SLUI was created for specific use, not all functionality will be beneficial for your application.  You should edit the source 
of SLUI to suit your needs.  However, alterted versions must be plainly marked as such.  View section 2 for licensing information.


2	== LICENSE ==

SLUI Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)

Copyright 2012 Matthew Overby. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1.	Redistributions of source code must retain the above copyright notice, this list of
	conditions and the following disclaimer.

2.	Redistributions in binary form must reproduce the above copyright notice, this list
	of conditions and the following disclaimer in the documentation and/or other materials
	provided with the distribution.

THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE UNIVERSITY OF MINNESOTA, DULUTH OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


3	== INSTALLATION AND USE ==

For information on how to use the SFGUI library, see "SampleBasic.cpp" in the "samples" directory.
SLUI was developed on Mac and Linux computers, and has not been tested in Microsoft Windows.

SLUI requires the programs:
1.	SFML 1.6  (http://www.sfml-dev.org)
2.	OpenGL  (http://www.opengl.org)
3.	CMake  (http://www.cmake.org)
4.	Boost  (http://www.boost.org)
5.	ncurses  (http://www.gnu.org/software/ncurses/ncurses.html)


4	== FEATURES ==

The features of SLUI are explained in greater detail in the sample files, located in the "samples" directory.
1.	General Controls:		General camera movement and mouse event handling.
2.	Click World:			Click on rendered objects and view information stored with these objects.
3.	Settings and Configuration:	Change options and settings, and store/load the changes.
4.	Console:			Textually view/change options and settings.
5.	Menu-Building Functions:	Functions that create/manage buttons and menus for easier programming and use.
6.	Graphing:			Plot and view data at runtime.


5	== TODO ==

There is some unfinished code and necessary cleanup in SLUI.  Due to time constraints, these were never completed:
1.	Make scrollbars draggable



