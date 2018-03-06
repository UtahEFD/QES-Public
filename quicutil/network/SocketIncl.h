#ifndef __SIVELAB_SOCKET_INCLS_H__
#define __SIVELAB_SOCKET_INCLS_H__

#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdio.h>

#pragma comment(lib, "Ws2_32.lib")

#else
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#endif

#endif
