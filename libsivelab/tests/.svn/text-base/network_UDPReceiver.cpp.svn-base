#include <iostream>
#include <string>

#include "util/handleNetworkArgs.h"
#include "network/UDPSocket.h"

using namespace sivelab;

int main(int argc, char *argv[])
{
  NetworkArgs nArgs;
  nArgs.process(argc, argv);

  // This code shows how to use either the IPv4 or the IPv6 sockets.
  // They cannot be interchanged in the current code, so only IPv4 -
  // IPv4 or IPv6 - IPv6 communication is supported.
  UDPSocket *uSock = 0;
  if (nArgs.domain == NetworkArgs::INET_IPv4)
    uSock = new UDPv4Socket(nArgs.port);
  else 
    uSock = new UDPv6Socket(nArgs.port);

  int count = 0;
  std::ostringstream d;
  while (count < 25)
    {
      uSock->recv( d );
      std::cout << "Got: " << d.str() << std::endl;
      count++;
    }

  uSock->close();
}
