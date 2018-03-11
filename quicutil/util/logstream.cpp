/*****************************************************************
 * logstream
 *
 * HANK Project, Copyright 1997-2000
 * The University of Iowa Computer Science Department
 * hank@cs.uiowa.edu
 *
 * Author: Pete Willemsen <willemsn@cs.uiowa.edu>
 *
 * Last Modification Info: $Id: logstream.cpp,v 1.1.1.1 2004/07/08 21:26:59 willemsn Exp $
 *****************************************************************/

// 
// NOTE: I learned about overriding streams from Dietmar Kuehl's web
// page.  If you're interested, the URL is (as of 12-14-99)
// http://www.informatik.uni-konstanz.de/~kuehl/c++/iostream/ Also,
// see "C++ IOStreams Handbook, by Steve Teale" as well as the C++
// Programming Language 3rd Ed book by Bjarne Stroustrup. -Pete Willemsen
//
// There is also now a good book on this subject.  Standard C++
// IOStreams and Locates by A. Langer and K. Kreft.

#include <unistd.h>

#include <iostream>
#include <cstdlib>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "logstream.h"

logbuf::logbuf()
  : streambuf(), 
  _opts_msgtype(los::status),
  _newline(true),
  _output_filedescr(1)
{
#ifndef SIVELAB_NO_OUTPUT
  _initialize_output_mask();
  _scan_environment_variables();
#endif
}

void logbuf::_determine_prefix( std::string& str )
{
#ifndef SIVELAB_NO_OUTPUT
  // Need to match priority levels to the current message stream.
  if (_opts_msgtype & los::status)
    str = "";
  else if (_opts_msgtype & los::debug1)
    str = "[DEBUG 1]: ";
  else if (_opts_msgtype & los::debug2)
    str = "[DEBUG 2]: ";
  else if (_opts_msgtype & los::debug3)
    str = "[DEBUG 3]: ";
  else if (_opts_msgtype & los::warning)
    str = "[WARNING]: ";
  else if (_opts_msgtype & los::error)
    str = "[ERROR]: ";
  else if (_opts_msgtype & los::fatal)
    str = "[FATAL]: ";
#endif
}

void logbuf::_scan_environment_variables( void )
{
#ifndef SIVELAB_NO_OUTPUT
  // //////////////////////////////////////////////////////////////////
  // Read the environment variable SIVELAB_MSG_DEBUG and determine what
  // value it has and set the _opts_msgmask accordingly.
  char *msg_debug_char = getenv(MSG_DEBUG);
  if (msg_debug_char) {
    int debug_lvl = (int) strtol( msg_debug_char, (char **)NULL, 10 );
    if ( debug_lvl == 1 )
      _opts_msgmask |= los::debug1;   // only print up to debug1 messages
    else if ( debug_lvl == 2 )
      _opts_msgmask |= los::debug1 | los::debug2;   // only print up to debug2 messages
    else if ( debug_lvl == 3 )
      _opts_msgmask |= los::debug1 | los::debug2 | los::debug3;   // only print up to debug3 messages
    else 
      // reset it to its default state showing all status, warning,
      // error, and fatal messages.
      _opts_msgmask = los::status | los::warning | los::error | los::fatal;
  }

  // //////////////////////////////////////////////////////////////////
  // We can print to a log file instead of to the terminal if
  // necessary.
#ifndef WIN32
  char *msg_fileio = getenv(MSG_FILE);
  if (msg_fileio) {
    _opts_output = los::use_file;

    // Extract the file name from the variable and attempt to write
    // it.  Need to perform error checking here!
    _output_filedescr = open(msg_fileio, O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR | S_IRGRP);
  } else {
    // The "file descriptor" is fixed if no file is specified.  We
    // will write the output stream to cout.
    _output_filedescr = 1;
  }
#endif

  // //////////////////////////////////////////////////////////////////
  // OVERRIDES ALL OTHER OPTIONS
  //
  // If this environment variable is detected, all output sent through
  // the logstream will be NOT be directed to any output stream!
  //
  if (getenv( MSG_OFF )) {
    _opts_output = los::om_noout;      // shuts off all output
  }
#endif
}

void logbuf::_initialize_output_mask( void ) 
{
#ifndef SIVELAB_NO_OUTPUT
  _opts_msgmask = los::status | los::warning | los::error | los::fatal;
#endif
}

void logbuf::_dispatch_errormsg( void )
{
#ifndef SIVELAB_NO_OUTPUT
  if ( _opts_msgtype & los::error )
    _error_cbptr( "\n\n**** logstream error callback request ****\n\n" );

  else if ( _opts_msgtype & los::fatal )
    _fatal_cbptr( "\n\n**** logstream fatal error callback request ****\n\n" );
#endif
}

int logbuf::_char_to_device( char c )
{
#ifndef SIVELAB_NO_OUTPUT
#ifndef WIN32
	if ( write(_output_filedescr, &c, 1) != 1 )
    return -1;
  else 
    return 0;
#else
	return 0;
#endif
#endif
}

// Global variable to hold the one logstream pointer
logstream* logstream::m_gbl_logstreamPtr = 0;

std::shared_ptr<logstream> logstream::instance()
{
  if (m_gbl_logstreamPtr == 0)
    {
      m_gbl_logstreamPtr = new logstream();
    }
  return std::shared_ptr<logstream>(m_gbl_logstreamPtr);
}

logstream::logstream() 
  : ostream(&_output_buf) {}

