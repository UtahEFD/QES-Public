/*
 *  ServerSocket.h
 *  NETCODE
 *
 *  Created by Pete Willemsen on 10/27/09.
 *  Copyright 2009 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 */

#ifndef __SIVELAB_SERVER_SOCKET_H__
#define __SIVELAB_SERVER_SOCKET_H__

#include <cstdio>
#include <sstream>

#include "util/logstream.h"

#include "IPAddress.h"
#include "Socket.h"
#include "ClientSocket.h"
#include "SocketException.h"

namespace sivelab {

  //!
  // ServerSocket
  // 
  class ServerSocket : public Socket
  {
  public:

    ServerSocket();
    ServerSocket(int port);

    virtual ~ServerSocket() { close(); }

    ClientSocket *accept() const;
    
#if 0
    virtual int send(IPAddress *sendTo_Address, const std::ostringstream& ostream) = 0;
    virtual int send(IPAddress *sendTo_Address, const void* data, const int sz) = 0;

    virtual int recv(std::ostringstream& ostream) = 0;
    virtual int recv(void* data, int sz) = 0;
#endif

  protected:
    int m_portNum;
    struct sockaddr_in6 m_servAddr;

    std::string m_serverIP;

  private:
  };

}


#endif
