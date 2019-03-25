/*****************************************************************
 * logstream
 *
 * Part of the SIVE Lab API and code base.
 * University of Minnesota Duluth
 * Pete Willemsen <willemsn@d.umn.edu>
 *
 * Originally, developed for the HANK Project, Copyright 1997-2000
 * The University of Iowa Computer Science Department
 * [willemsn@cs.uiowa.edu]
 *
 * Author: Pete Willemsen <willemsn@cs.uiowa.edu>
 *
 * Last Modification Info: $Id: logstream.h,v 1.1.1.1 2004/07/08 21:26:59 willemsn Exp $
 *****************************************************************/

#ifndef __LOG_STREAMBUFFER_H__
#define __LOG_STREAMBUFFER_H__

#include <cstdio>
#include <iostream>
#include <string>
#ifndef WIN32
#include <tr1/memory>
#endif

// Goal of these classes: act as single mechanism through which output
// can be sent (originally written for the Hank simulator at Iowa).
// The log-class can be tailored to output to different devices.  By
// default, it outputs to the cout stream.  Additionally, the logger
// can send output to a specified log file.  Also, output can be
// suspended, eliminated, or compiled out using environmental
// variables, compiler defines, or state variables.  The important
// functions are inlined so that maximum efficiency can be gained when
// the output is not compiled.  Priority levels can be specified to
// accomodate different levels of error messages being generated
// by the simulation software.

#define ENVVAR_PREFIX "SIVELAB"
#define MSG_FILE      ENVVAR_PREFIX "_MSG_FILE"
#define MSG_OFF       ENVVAR_PREFIX "_MSG_OFF"
#define MSG_DEBUG     ENVVAR_PREFIX "_MSG_DEBUG"

#define _LS_CLEAR    0
#define _LS_STATUS   1
#define _LS_DEBUG1   2
#define _LS_DEBUG2   4
#define _LS_DEBUG3   8
#define _LS_WARNING 16
#define _LS_ERROR   32
#define _LS_FATAL   64
#define _LS_NOOUT  128

#define _LS_FILE 1

// ///////////////////////////////////////////////////////////////////
// class los - specified the state of the message being sent to the
// logstream.  "status" message are the default type of messages and
// should be used for general status messages.  "warning", "error",
// and "fatal" reflect the seriousness of the log message.
class los {
public:
  enum msg_type {
    ml_clear = _LS_CLEAR,
    status = _LS_STATUS,   // status level message
    debug1 = _LS_DEBUG1,   // debugging message level 1
    debug2 = _LS_DEBUG2,   // debugging message level 2
    debug3 = _LS_DEBUG3,   // debugging message level 3
    warning = _LS_WARNING, // warning message
    error = _LS_ERROR,     // general error
    fatal = _LS_FATAL      // fatal error        
  };

  enum output_mode {
    om_clear = _LS_CLEAR,
    om_noout = _LS_NOOUT,
    use_file = _LS_FILE
  };
};


// ///////////////////////////////////////////////////////////////////
// class logbuf - this class performs all the work to redirect and
// label the output stream.  It recognizes various environment
// variables to tailor the output.
using namespace std;
class logbuf : public streambuf
{
public:
  logbuf();

  void setMsgType( los::msg_type l ); // inlined function - see below
  
  void error_callback( void (*f_ptr)( const std::string& ) ) { _error_cbptr = f_ptr; }
  void fatal_callback( void (*f_ptr)( const std::string& ) ) { _fatal_cbptr = f_ptr; }

protected:
  int overflow( int c=EOF );   // inlined function - see below

private:
  los::msg_type    _opts_msgtype;   // determines what gets printed
  int              _opts_msgmask;   // 
  los::output_mode _opts_output;

  bool _newline;
  
  int _output_filedescr;

  void (*_fatal_cbptr)( const std::string& );
  void (*_error_cbptr)( const std::string& );

  void _determine_prefix( std::string& str );
  void _scan_environment_variables( void );
  void _initialize_output_mask( void );
  void _dispatch_errormsg( void );
  int _char_to_device( char c );
  
  logbuf( const logbuf& );
  logbuf& operator=( const logbuf& );
};


// ///////////////////////////////////////////////////////////////////
// class logstream - the stream class that uses the logbuf stream
// buffer.
class logstream : public ostream 
{
public:
  
  //
  // Static function for accessing the one instance of an object
  // factory... this should really be a shared_ptr.
  //
#ifdef WIN32
	static logstream* instance();
#else
	static std::tr1::shared_ptr<logstream> instance();
#endif

  void setType( los::msg_type mt );  // inlined function - see below

  // Provide means for signal handling functions to be called upon
  // error or fatal error messages.
  void registerErrorCallback( void (*f_ptr)( const std::string& ) ) { _output_buf.error_callback( f_ptr ); }
  void registerFatalErrorCallback( void (*f_ptr)( const std::string& ) ) { _output_buf.fatal_callback( f_ptr ); }

protected:
  // Should be private since I do not WANT instances of these to be created yet... or do I?
  logstream();

private:

  logbuf _output_buf;

  logstream( const logstream& );
  logstream& operator=( const logstream& );

  static logstream* m_gbl_logstreamPtr;
};


inline void logbuf::setMsgType( los::msg_type l ) 
{ 
#ifndef SIVELAB_NO_OUTPUT
  _opts_msgtype = l; 
#endif
}

inline int logbuf::overflow( int c ) 
{ 
#ifndef SIVELAB_NO_OUTPUT
  if ((_opts_output != los::om_noout) && ((_opts_msgtype & _opts_msgmask) > 0)) {
    
    if (c == '\n')
      _newline = true;
    else if (_newline) {
      _newline = false;  // reset the newline marker
      
      std::string prefix;
      _determine_prefix( prefix );
      int len = prefix.length();
      
      if (len > 0) {
	if (sputn(prefix.c_str(), len) != len) 
	  return EOF;
      }
    }
    
    if (_char_to_device(char(c)) > 0)
      return EOF;
    else
      return c;
  } 
  else
    return 0;
#else
  // NO OUTPUT, JUST RETURN
  return 0;
#endif

 return 1;
}

inline void logstream::setType( los::msg_type mt )
{ 
#ifndef SIVELAB_NO_OUTPUT
  _output_buf.setMsgType( mt ); 
#endif
}

extern logstream hanklog;

#endif
