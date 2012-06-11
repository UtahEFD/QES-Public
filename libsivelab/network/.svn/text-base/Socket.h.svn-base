/*
 *  Socket.h
 *  NETCODE
 *
 *  Created by Pete Willemsen on 10/21/09.
 *  Copyright 2009 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 */

#ifndef __SOCKET_H__
#define __SOCKET_H__ 1

#include <cerrno>
#include <string>
#include "SocketIncl.h"

namespace sivelab {

  class Socket
  {
  public:
    Socket();
    virtual ~Socket();

#if 0
    // data sending should be abstracted into this class if possible....
    send
    recv

    operator<<
    operator>>
#endif

    virtual int send( std::string &msg );
    virtual int receive( std::string &msg );

    // given a string with a host name, determine if it's a IP address
    // string or a hostname
    static void testhost(const std::string &host)
    {
      // first, check to see if the first character is a digit
      

      // or count the number of periods
      
      // if it contains some number of :, has to be IPv6 hostip string
      
    }

    virtual int domain() { return m_family; }
    void close() 
	{
#ifdef WIN32
		closesocket(m_socket_fd);
#else
		::close(m_socket_fd);
#endif
	}
  protected:

#ifdef WIN32
    SOCKET m_socket_fd;
#else
    int m_socket_fd;
#endif

    int m_family;
  };

}

#endif // #define __SOCKET_H__ 1

