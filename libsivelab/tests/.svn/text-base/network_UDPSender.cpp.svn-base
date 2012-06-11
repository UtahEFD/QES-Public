#include <iostream>
#include <string>

#include <signal.h>

#include "util/handleNetworkArgs.h"
#include "network/UDPSocket.h"

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

  UDPSocket *uSock = 0;
  if (nArgs.domain == NetworkArgs::INET_IPv4)
    uSock = new UDPv4Socket();
  else 
    uSock = new UDPv6Socket();
  
  // Receiver ip address and port number for socket address
  IPAddress *sendTo_Addr = 0;
  if (nArgs.domain == NetworkArgs::INET_IPv4)
    {
      assert(nArgs.hasHostname || nArgs.hasHostIP);

      sendTo_Addr = new IPv4Address(); 

      if (nArgs.hasHostname)
	sendTo_Addr->setHostname(nArgs.hostname, nArgs.port);
      else
	sendTo_Addr->setIPAddrStr(nArgs.hostIP, nArgs.port);
    }
  else 
    sendTo_Addr = new IPv6Address(nArgs.hostname, nArgs.port);

  std::ostringstream d("");
  int count = 0;
  while (shutdownSender != true)
    {
      d.str("");
#ifdef WIN32
	  d << count << ": " << 2;
#else
	  d << count << ": " << time(NULL);
#endif
	  uSock->send(sendTo_Addr, d);
      count++;

#ifndef WIN32
	  usleep(100000);
#endif
  }
  std::cout << "Sent " << count << " messages via UDP." << std::endl;

  uSock->close();
}
