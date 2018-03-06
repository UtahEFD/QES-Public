/*!
 *  IPAddress.h
 *  SIVE-network
 *  /brief brief description test
 *
 *  Created by Pete Willemsen on 10/27/09.
 *  Copyright 2009 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 */

#ifndef __IPADDRESS_H__
#define __IPADDRESS_H__ 1

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <string>
#include <sys/types.h>

#include "SocketIncl.h"

namespace sivelab {

  // abstracts the sockaddr structures, and provides DNS conversions
  // for IPv4 and IPv6 addresses.
  class IPAddress
  {
  public:
    IPAddress();
    virtual ~IPAddress();

    virtual void setIPAddrStr(const std::string& ipAddrStr, int port = 0) = 0;
    virtual void setHostname(const std::string& hostname, int port = 0) = 0;
    
    virtual int family(void) const = 0;

    virtual socklen_t length(void) const = 0;

    virtual const struct sockaddr *genericAddrPtr(void) const = 0;

    std::string getHostname() { return m_hostname; }
    std::string getIPAddrString() { return m_ipAddrStr; }

    
    // this is slightly bad since the port doesn't get reflected in the actual socket address structure...!!!
    void setPortNum(const int port) {m_port = port;}
    int port(void) const {return m_port;}
    
  protected:
    std::string m_ipAddrStr;
    std::string m_hostname;

    int m_port;

    struct sockaddr_storage m_sockAddr;
  private:
  };

  class IPv4Address : public IPAddress
  {
  public:
    IPv4Address();
    IPv4Address(const std::string &hostname, int port=0);
    virtual ~IPv4Address();
    
    void setIPAddrStr(const std::string& ipAddrStr, int port = 0);
    void setHostname(const std::string& hostname, int port = 0);

    int family(void) const { return AF_INET; }
    socklen_t length(void) const { return sizeof(struct sockaddr_in); }
    const struct sockaddr *genericAddrPtr(void) const { return (const struct sockaddr *)m_sockAddr4; }

#ifdef WIN32
    int getHostAddress() { return m_sockAddr4->sin_addr.s_addr; }
#else
    in_addr_t getHostAddress() { return m_sockAddr4->sin_addr.s_addr; }
#endif

  private:
    struct sockaddr_in *m_sockAddr4;
  };

  class IPv6Address : public IPAddress
  {
  public:
    IPv6Address();
    IPv6Address(const std::string &hostname, int port=0);
    virtual ~IPv6Address();
    
    void setIPAddrStr(const std::string& ipAddrStr, int port = 0);
    void setHostname(const std::string& hostname, int port = 0);

    int family(void) const { return AF_INET6; }
    socklen_t length(void) const { return sizeof(struct sockaddr_in6); }
    const struct sockaddr *genericAddrPtr(void) const { return (struct sockaddr *)m_sockAddr6; }

    in6_addr getHostAddress() { return m_sockAddr6->sin6_addr; }

  private:
    struct sockaddr_in6 *m_sockAddr6;
  };
}

#endif // __IPADDRESS_H__ 1


