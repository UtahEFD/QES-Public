#include <iostream>
#include <string>

#include "util/handleNetworkArgs.h"
#include "network/UDPSocket.h"

#include "netData.h"

using namespace sivelab;

int main(int argc, char *argv[])
{
  NetworkArgs nArgs;
  nArgs.process(argc, argv);

  assert(nArgs.domain == NetworkArgs::INET_IPv4);
  UDPSocket *uSock = new UDPv4Socket(nArgs.port);

  int count = 0;
  netData nd;
  while (count < 25)
    {
      uSock->recv( &nd, sizeof(netData) );
      std::cout << "Got: " << nd.i1 << ", " << nd.l1 << ", " << nd.f1 << ", " << nd.d1 << std::endl;
      count++;
    }

  uSock->close();
}
