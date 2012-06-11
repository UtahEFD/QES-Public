#include <cassert>
#include <iostream>
#include <string>

#include <signal.h>

#include "util/handleNetworkArgs.h"
#include "network/UDPSocket.h"

#include "netData.h"

using namespace sivelab;

static bool shutdownSender = false;

void signalHandler(int sig)
{
  std::cout << "Signal Caught!" << std::endl;
  shutdownSender = true;
}

int main(int argc, char *argv[])
{
  signal(SIGINT, signalHandler);

  NetworkArgs nArgs;
  nArgs.process(argc, argv);

  assert(nArgs.domain == NetworkArgs::INET_IPv4);
  UDPSocket *uSock = new UDPv4Socket();
  
  // Receiver ip address and port number for socket address
  assert(nArgs.hasHostname || nArgs.hasHostIP);
  IPAddress *sendTo_Addr = new IPv4Address(); 

  if (nArgs.hasHostname)
    sendTo_Addr->setHostname(nArgs.hostname, nArgs.port);
  else
    sendTo_Addr->setIPAddrStr(nArgs.hostIP, nArgs.port);

  std::ostringstream d("");
  int count = 0;
  netData nd;
  nd.i1 = 0;
  nd.l1 = 0;
  nd.f1 = 0.0;
  nd.d1 = 0.0;
  while (shutdownSender != true)
    {
      // generate data each frame and send
      nd.i1 += 2;
      nd.l1 = count;
      nd.f1 += 0.01f;
      nd.d1 += 0.0025f;

      uSock->send(sendTo_Addr, &nd, sizeof(netData));
      count++;

#ifdef WIN32
#else
	  usleep(100000);
#endif 
  }
  std::cout << "Sent " << count << " messages via UDP." << std::endl;

  uSock->close();
}
