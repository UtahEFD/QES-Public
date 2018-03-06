/*
 *  IPAddress.cpp
 *  NETCODE
 *
 *  Created by Pete Willemsen on 10/27/09.
 *  Copyright 2009 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 */

#include "IPAddress.h"

using namespace sivelab;

IPAddress::IPAddress()
{
}

IPAddress::~IPAddress()
{
}

IPv4Address::IPv4Address()
{
  memset(&m_sockAddr, 0, sizeof(struct sockaddr_in));
  m_sockAddr4 = (struct sockaddr_in*)&m_sockAddr;
}

IPv4Address::IPv4Address(const std::string &hostname, int port)
{
  m_hostname = hostname;
  m_port = port;

  memset(&m_sockAddr, 0, sizeof(struct sockaddr_in));
  m_sockAddr4 = (struct sockaddr_in*)&m_sockAddr;

  setHostname(hostname, m_port);
}

IPv4Address::~IPv4Address()
{
}

void IPv4Address::setHostname(const std::string& hostname, int port)
{
  m_hostname = hostname;
  m_port = port;

  // If we were given a hostname it is possible to take the "res"
  // result from getaddrinfo as the server address we wish to
  // connect to.  However, in this example, we use getaddrinfo +
  // inet_ntop to get the ip address string.  We will then use this
  // string with inet_pton to convert back.  It is excessive, but it
  // hopefully shows you what is happening.

  struct addrinfo hints, *res0;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  // hints.ai_socktype = SOCK_STREAM;  // doesn't really matter for this... 
  // we just want to convert the hostname to something useful

  getaddrinfo(m_hostname.c_str(), 0, 0, &res0);

  // set some decent things in here
  m_sockAddr4->sin_family = AF_INET;   // only IPv4 for now
  m_sockAddr4->sin_port = htons(m_port);

  char ipAddrStr[1024];
  struct sockaddr_in *tmpAddr = (struct sockaddr_in*)(res0->ai_addr);

  m_sockAddr4->sin_addr = tmpAddr->sin_addr;

#ifdef WIN32
  inet_ntop(res0->ai_family, (PVOID)&(tmpAddr->sin_addr), ipAddrStr, 1024);
#else
  inet_ntop(res0->ai_family, (const void *)&(tmpAddr->sin_addr), ipAddrStr, 1024);      
#endif
  m_ipAddrStr = ipAddrStr;
}

void IPv4Address::setIPAddrStr(const std::string& ipAddrStr, int port)
{
  m_ipAddrStr = ipAddrStr;
  m_port = port;

  // set some decent things in here
  memset(&m_sockAddr, 0, sizeof(struct sockaddr_storage));
  // m_sockAddr4->sin_len = sizeof(struct sockaddr_in);
  m_sockAddr4->sin_family = AF_INET;
  m_sockAddr4->sin_port = htons(m_port);

  std::cout << "m_ipAddr = " << m_ipAddrStr.c_str() << std::endl;
  if (inet_pton(AF_INET, m_ipAddrStr.c_str(), &(m_sockAddr4->sin_addr)) != 1)
    {
      std::cout << "bad inet_pton conversion" << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << "sin_addr = " << ntohl(m_sockAddr4->sin_addr.s_addr) << std::endl;
    
  char hostName[NI_MAXHOST];

  int gni_error_code = getnameinfo((struct sockaddr *)m_sockAddr4, sizeof(struct sockaddr_in), hostName, sizeof(hostName), NULL, 0, NI_NAMEREQD);

  if (gni_error_code != 0) 
    {
      std::cerr <<  "could not resolve hostname: (" << gni_error_code << ") " << gai_strerror(gni_error_code) << std::endl;
      
      exit(EXIT_FAILURE);
    }

  //  error = getnameinfo((struct sockadddr *)&ss, ss.ss_len,
  //			    namebuf, sizeof(namebuf), NULL, 0, 0);
//	if (error)
//		perror("getnameinfo");

  m_hostname = hostName;
}



IPv6Address::IPv6Address()
{
  memset(&m_sockAddr, 0, sizeof(struct sockaddr_in6));
  m_sockAddr6 = (struct sockaddr_in6*)&m_sockAddr;
}

IPv6Address::IPv6Address(const std::string &hostname, int port)
{
  m_hostname = hostname;
  m_port = port;

  memset(&m_sockAddr, 0, sizeof(struct sockaddr_in6));
  m_sockAddr6 = (struct sockaddr_in6*)&m_sockAddr;

  setHostname(hostname, m_port);
}


IPv6Address::~IPv6Address()
{
}

void IPv6Address::setHostname(const std::string& hostname, int port)
{
  m_hostname = hostname;
  m_port = port;

  // If we were given a hostname it is possible to take the "res"
  // result from getaddrinfo as the server address we wish to
  // connect to.  However, in this example, we use getaddrinfo +
  // inet_ntop to get the ip address string.  We will then use this
  // string with inet_pton to convert back.  It is excessive, but it
  // hopefully shows you what is happening.

  struct addrinfo hints, *res0, *currRes;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET6;
  hints.ai_socktype = 0; // SOCK_STREAM;  // doesn't really matter for
				    // this... we just want to convert
				    // the hostname to something
				    // useful
  
  getaddrinfo(m_hostname.c_str(), 0, &hints, &res0);
  assert(res0);

  for(currRes = res0; currRes != 0; currRes = currRes->ai_next)
    {
      if (currRes->ai_family == AF_INET6)
	{
	  // set some decent things in here
	  m_sockAddr6->sin6_family = AF_INET6;
	  m_sockAddr6->sin6_port = htons(m_port);
	  
	  char ipAddrStr[1024];
	  struct sockaddr_in6 *tmpAddr6 = (struct sockaddr_in6*)(res0->ai_addr);
	  m_sockAddr6->sin6_addr = tmpAddr6->sin6_addr;
	  
	  inet_ntop(AF_INET6, &(tmpAddr6->sin6_addr), ipAddrStr, 1024);      
	  m_ipAddrStr = ipAddrStr;

	  break;
	}
    }
}

void IPv6Address::setIPAddrStr(const std::string& ipAddrStr, int port)
{
  m_ipAddrStr = ipAddrStr;
  m_port = port;

  // set some decent things in here
  // bzero(m_sockAddr4, sizeof(struct sockaddr_in));
  m_sockAddr6->sin6_family = AF_INET6;
  m_sockAddr6->sin6_port = htons(m_port);

  if (inet_pton(AF_INET6, m_ipAddrStr.c_str(), &(m_sockAddr6->sin6_addr)) == -1)
    {
      std::cout << "bad inet_pton conversion" << std::endl;
      exit(EXIT_FAILURE);
    }
    
  char hostName[NI_MAXHOST];
  if (getnameinfo((const struct sockaddr*)&m_sockAddr, sizeof(struct sockaddr_in6), hostName, NI_MAXHOST, NULL, 0, NI_NAMEREQD)) 
    {
      std::cerr <<  "could not resolve hostname" << std::endl;
    }

  m_hostname = hostName;
}
