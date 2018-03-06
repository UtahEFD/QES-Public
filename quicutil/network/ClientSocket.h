/*
 *  ClientSocket.h
 *  NETCODE
 *
 *  Created by Pete Willemsen on 10/27/09.
 *  Copyright 2009 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 */

#ifndef __SIVELAB_CLIENT_SOCKET_H__
#define __SIVELAB_CLIENT_SOCKET_H__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include "IPAddress.h"
#include "Socket.h"
#include "SocketException.h"
#include "ProtocolMessageBuffer.h"

namespace sivelab {

  //!
  // ClientSocket
  // 
  class ClientSocket : public Socket
  {
  public:

    ClientSocket();
    ClientSocket(const std::string &host, int port);

    ClientSocket(int socketfd, struct sockaddr_in6 &addr);

    virtual ~ClientSocket() { close(); }

    int send( std::string &msg );
    int receive( std::string &msg );

    /** @brief Socket data sending
     * 
     * Function that sends a message buffer.
     */
    int send( ProtocolMessageBuffer &msg );

    /** @brief Socket data receiving
     * 
     * Function that receives data on the socket and can either block
     * or not block.  Blocks by default, which is typical socket
     * behavior.
     */
    int receive( ProtocolMessageBuffer &msg, bool allowBlocking=true );

    std::string peername();
    std::string sockname();

#if 0
    virtual int send(IPAddress *sendTo_Address, const std::ostringstream& ostream) = 0;
    virtual int send(IPAddress *sendTo_Address, const void* data, const int sz) = 0;

    virtual int recv(std::ostringstream& ostream) = 0;
    virtual int recv(void* data, int sz) = 0;
#endif

  protected:
    int m_portNum;
    struct sockaddr_in6 m_clientAddr;

    std::string m_sockIP;
    std::string m_peerIP;

  private:
  };

}


#endif
