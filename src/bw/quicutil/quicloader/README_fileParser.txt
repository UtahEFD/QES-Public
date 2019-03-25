===========================================================

README for inputFileParser (ifp)

author: Andrew Larson <lars2865@d.umn.edu>
author: Michael Jacobson <jacob970@d.umn.edu>

created: Sat Apr 18 17:55:27 CDT 2009

purpose: 
To provide a basic framework for getting data in and out 
of files using C++, where tagging and parsing can be 
defined on an application by application basis, but the
parser is not part integrated into the parser.

sections:

	1) GOALS
	
	2) TODO
	
	3) FILES

===========================================================
#



GOALS
==========================================================

* Goal 1: implement a standard file parser that
	emulates the current capability of the original ifp. 
	- Nearing completion.

* Goal 2: implement a standardized QUICifp.

* Goal 3: implement a UML QUICifp.
	-> Move to a seperate directory
	
#



TODO
===========================================================

** Figure out recalls. How's this going to happen?????

* Define a standard (default) file interpretation scheme.
	This is needed for the base classes of attributes and 
	tag. 
	- Started. Needs verification
		
*	Do documentation to make Pete happy and the lite reusable

* Think up some good testing schemes. Look for mem leaks.
	- Test datamList for add, remove, copy and delete
	- Test attributList for add, remove, copy and delete
	- Test Tag and Attribute for copy and delete

* Implement some good testing schemes. Look for mem leaks.

* Figure out Typing scheme. How to force on derived class?
	- commit takes an int now.
	- Types can be defined in element child.

#



FILES
===========================================================

fileParser
 |
 +-standard
 |  | sfpTester.cpp
 |	| standardElements.h
 |	| standardElements.cpp
 |	| standardFileParser.h
 |	| standardFileParser.cpp
 |  \ tester.txt
 |
 +-xml
 |
 | Makefile
 | README.txt
 | attribute.h
 | attributeList.h
 | datam.h
 | datamList.h
 | element.h
 | fileParser.h
 | tag.h
 | attribute.cpp
 | attributeList.cpp
 | datam.cpp
 | datamList.cpp
 | fileParser.cpp
 \ tag.cpp

#




