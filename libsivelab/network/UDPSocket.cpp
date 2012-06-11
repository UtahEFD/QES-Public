/*
 *  UDPSocket.cpp
 *  NETCODE
 *
 *  Created by Pete Willemsen on 10/27/09.
 *  Copyright 2009 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 */

#include "UDPSocket.h"

using namespace sivelab;

UDPSocket::UDPSocket() {}

void UDPSocket::p_createSocket()
{
  if ((m_socket_fd = socket(m_family, SOCK_DGRAM, 0)) < 0)
    {
      std::cerr << "UDP socket - socket creation error" << std::endl;
      // just die for now...
      exit(EXIT_FAILURE);
    }
}

void UDPSocket::p_bindSenderInfo(const struct sockaddr* addr, socklen_t addrsz)
{
  int err = -1;
  err = bind(m_socket_fd, addr, addrsz);
  
  if (err)
    {
      std::cerr << "BIND error!!!" << std::endl;
      exit(EXIT_FAILURE);
    }
}

#if 0
void UDPSocket::cacheReceiverInfo(IPAddress *sockAddrInfo)
{
  m_receiverAddr = sockAddrInfo;
}
#endif

#if 0
int UDPSocket::send(IPAddress *sendTo_Addr, const std::ostringstream& sockStream)
{
  std::cout << "UDPSocket::send --> sendTo = [" << sendTo_Addr->getIPAddrString() << "], port=" << sendTo_Addr->port() << std::endl;

  int sentBytes = 0;
  sentBytes = sendto(m_socket_fd, 
		     sockStream.str().c_str(), 
		     sockStream.str().length(), 
		     0, 
		     (const struct sockaddr*)sendTo_Addr->genericAddrPtr(), 
		     sendTo_Addr->length());

  if (sentBytes != (ssize_t)sockStream.str().length())
    {
      std::cerr << "write, sendto error, sentBytes=" << sentBytes << std::endl;
      perror("error");
    }
  return sentBytes;
}
#endif




UDPv4Socket::UDPv4Socket(int portNum)
  : UDPSocket()
{
  m_family = AF_INET;
  m_portNum = portNum;
  p_createSocket();

  p_initializeSocket();
}

void UDPv4Socket::p_initializeSocket()
{
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));

  addr.sin_family = m_family;
  addr.sin_port = htons(m_portNum);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);

  p_bindSenderInfo((const struct sockaddr*)&addr, sizeof(addr));
}

UDPv4Socket::~UDPv4Socket() {}


int UDPv4Socket::send(IPAddress *sendTo_Addr, const std::ostringstream& sockStream)
{
  std::cout << "UDPv4Socket::send --> sendTo = [" << sendTo_Addr->getIPAddrString() << "], port=" << sendTo_Addr->port() << std::endl;

  int sentBytes = 0;
  sentBytes = sendto(m_socket_fd, 
		     sockStream.str().c_str(), 
		     sockStream.str().length(), 
		     0, 
		     (const struct sockaddr*)sendTo_Addr->genericAddrPtr(), 
		     sendTo_Addr->length());

  if (sentBytes != sockStream.str().length())
    {
      std::cerr << "write, sendto error, sentBytes=" << sentBytes << std::endl;
      perror("error");
    }
  return sentBytes;
}

int UDPv4Socket::send(IPAddress *sendTo_Addr, const void *data, const int sz)
{
  std::cout << "UDPv4Socket::send --> sendTo = [" << sendTo_Addr->getIPAddrString() << "], port=" << sendTo_Addr->port() << std::endl;

  int sentBytes = 0;
  sentBytes = sendto(m_socket_fd,
#ifdef WIN32
	  (const char*) data,
#else
		     data,
#endif
		     sz,
		     0, 
		     (const struct sockaddr*)sendTo_Addr->genericAddrPtr(), 
		     sendTo_Addr->length());

  if (sentBytes != sz)
    {
      std::cerr << "write, sendto error, sentBytes=" << sentBytes << std::endl;
      perror("error");
    }
  return sentBytes;
}


int UDPv4Socket::recv(std::ostringstream& sockStream)
{
  // std::cout << "UDPIPv4Socket::recv --> " << std::endl;

  int nbytes = 1024;
  char data[1024];
  int recvBytes = -1;

  // wait for receiver to send something back...
  struct sockaddr_in recvdAddress;
  socklen_t recvdAddress_len = sizeof(recvdAddress);
  memset(&recvdAddress, 0, sizeof(recvdAddress));
  
  if ((recvBytes = recvfrom(m_socket_fd, data, nbytes, 0, (struct sockaddr*)&recvdAddress, &recvdAddress_len)) < 0)
    {
      std::cerr << "read, recvfrom error" << std::endl;
    }

#if 0
  else if (m_family == AF_INET6)
    {
      struct sockaddr_in6 recvdAddress;
      socklen_t recvdAddress_len = sizeof(sockaddr_in6);
      bzero(&recvdAddress, sizeof(sockaddr_in6));
      

      if ((recvBytes = recvfrom(m_socket_fd, data, nbytes, 0, (struct sockaddr*)&recvdAddress, &recvdAddress_len)) < 0)
	{
	  std::cerr << "read, recvfrom error" << std::endl;
	}
    }
#endif

  sockStream.str(data);
  return recvBytes;
}



int UDPv4Socket::recv(void* data, int sz)
{
  // std::cout << "UDPIPv4Socket::recv --> " << std::endl;

  int recvBytes = -1;

  // wait for receiver to send something back...
  struct sockaddr_in recvdAddress;
  socklen_t recvdAddress_len = sizeof(recvdAddress);
  memset(&recvdAddress, 0, sizeof(recvdAddress));
  
  if ((recvBytes = recvfrom(m_socket_fd, 
#ifdef WIN32
	  (char *)data,
#else
	  data,
#endif
	  sz, 0, (struct sockaddr*)&recvdAddress, &recvdAddress_len)) < 0)
    {
      std::cerr << "read, recvfrom error" << std::endl;
    }

#if 0
  else if (m_family == AF_INET6)
    {
      struct sockaddr_in6 recvdAddress;
      socklen_t recvdAddress_len = sizeof(sockaddr_in6);
      bzero(&recvdAddress, sizeof(sockaddr_in6));
      

      if ((recvBytes = recvfrom(m_socket_fd, data, nbytes, 0, (struct sockaddr*)&recvdAddress, &recvdAddress_len)) < 0)
	{
	  std::cerr << "read, recvfrom error" << std::endl;
	}
    }
#endif

  return recvBytes;
}




UDPv6Socket::UDPv6Socket(int portNum)
  : UDPSocket()
{
  m_family = AF_INET6;
  m_portNum = portNum;
  p_createSocket();

  p_initializeSocket();
}

void UDPv6Socket::p_initializeSocket()
{
  struct sockaddr_in6 addr;
  memset(&addr, 0, sizeof(addr));

  addr.sin6_family = m_family;
  addr.sin6_port = htons(m_portNum);
  addr.sin6_addr = in6addr_any;

  p_bindSenderInfo((const struct sockaddr*)&addr, sizeof(addr));
}

UDPv6Socket::~UDPv6Socket() {}

#if 0
int UDPSocket::send(const std::ostringstream& sockStream)
{
  int sentBytes = 0;
  socklen_t m_receiverAddrLen = sizeof(m_receiverAddr);

  sentBytes = sendto(m_socket_fd, sockStream.str().c_str(), sockStream.str().length(), 0, (struct sockaddr*)&m_receiverAddr, m_receiverAddrLen);
  if (sentBytes != (ssize_t)sockStream.str().length())
    {
      std::cerr << "write, sendto error, sentBytes=" << sentBytes << std::endl;
      perror("error");
    }
  return sentBytes;
}
#endif
