/*
 *  UDPSocket.h
 *  NETCODE
 *
 *  Created by Pete Willemsen on 10/27/09.
 *  Copyright 2009 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 */

#ifndef __UDPSOCKET_H__
#define __UDPSOCKET_H__ 1

#include <cstdio>
#include <sstream>
#include "IPAddress.h"
#include "Socket.h"

namespace sivelab {

  //!
  // UDPSocket
  // 
  class UDPSocket : public Socket
  {
  public:

    UDPSocket();
    virtual ~UDPSocket() { close(); }

    virtual int send(IPAddress *sendTo_Address, const std::ostringstream& ostream) = 0;
    virtual int send(IPAddress *sendTo_Address, const void* data, const int sz) = 0;

    virtual int recv(std::ostringstream& ostream) = 0;
    virtual int recv(void* data, int sz) = 0;

    // void cacheReceiverInfo(IPAddress *sockAddrInfo);
    // overload the operator<< and operator>>

  protected:
    int m_portNum;

    void p_createSocket();
    void p_bindSenderInfo(const struct sockaddr* addr, socklen_t addrsz);

  private:
  };

  //!
  // UDPv4Socket
  // 
  class UDPv4Socket : public UDPSocket
  {
  public:
    UDPv4Socket(int portNum=0);
    ~UDPv4Socket();

    int domain() { return AF_INET; }
    
    int send(IPAddress *sendTo_Address, const std::ostringstream& ostream);
    int send(IPAddress *sendTo_Address, const void *data, const int sz);

    int recv(std::ostringstream& ostream);
    int recv(void* data, int sz);

  private:
    void p_initializeSocket();
	  
	  IPv4Address m_senderAddr, m_receiverAddr;
  };

  //!
  // UDPv6Socket
  // 
  class UDPv6Socket : public UDPSocket
  {
  public:
    UDPv6Socket(int portNum=0);
    ~UDPv6Socket();

    int domain() { return AF_INET6; }
    int send(IPAddress *sendTo_Address, const std::ostringstream& ostream) { return 0; }
    int send(IPAddress *sendTo_Address, const void* data, const int sz) { return 0; }
    int recv(std::ostringstream& ostream) { return 0; }
    int recv(void* data, int sz) { return 0; }

  private:
    void p_initializeSocket();
	  
	  IPv6Address m_senderAddr, m_receiverAddr;
  };

}

#endif
