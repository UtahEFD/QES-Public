#include <iostream>
#include <string>

#include "util/handleNetworkArgs.h"
#include "network/IPAddress.h"

using namespace sivelab;

int main(int argc, char *argv[])
{
  NetworkArgs args;
  args.process(argc, argv);

  if (args.verbose)
    {
      if (args.hasHostname)
	{
	  std::cout << "Converting hostname (" << args.hostname << ") to host IP address." << std::endl;
	  IPv4Address ip;
	  ip.setHostname( args.hostname );
	  std::cout << "\t" << args.hostname << " is " << ip.getIPAddrString() << std::endl;
	}

      if (args.hasHostIP)
	{
	  std::cout << "Converting host IP address (" << args.hostIP << ") to hostname." << std::endl;
	  IPv4Address ip;
	  ip.setIPAddrStr( args.hostIP );
	  std::cout << "\t" << args.hostIP << " is " << ip.getHostname() << std::endl;
	}
    }

#if 0
  // 
  // Test IPv4 conversions
  //
  std::string test_ipv4Str = "131.212.7.1";
  std::string test_hostStr = "ilmatar.d.umn.edu";

  IPv4Address ipv4_00;
  ipv4_00.setIPAddrStr(test_ipv4Str);
  std::cout << "IP Addr (" << test_ipv4Str << ") is " << ipv4_00.getHostname() << std::endl;
  
  IPv4Address ipv4_01;
  ipv4_01.setHostname( test_hostStr );
  std::cout << "Host (" << test_hostStr << ") in dotted decimal [" << ipv4_01.getIPAddrString() << "]" << std::endl;
#endif

#if 0
  // 
  // Test IPv6 conversions
  //
  std::string test_ipv6Str = "2001:468:1920:7:a6ba:dbff:fefb:5881";
  test_hostStr = "ipv6.google.com";

  IPv6Address ip6_00;
  ip6_00.setHostname(test_hostStr);
  std::cout << "Host (" << test_hostStr << ") in IPv6 [" << ip6_00.getIPAddrString() << "]" << std::endl;

  IPv6Address ip6_01;
  ip6_01.setIPAddrStr(test_ipv6Str);
  std::cout << ip6_01.getIPAddrString() << std::endl;
  std::cout << "IPv6 Addr (" << test_ipv6Str << ") in IPv6 [" << ip6_01.getHostname() << "]" << std::endl;
#endif
}
