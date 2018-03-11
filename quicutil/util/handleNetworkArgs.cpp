/*
 *  handleNetworkArgs.cpp
 *
 *  Created by Pete Willemsen on 10/6/09.
 *  Copyright 2009 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 * This file is part of libSIVELab (libsivelab).
 *
 * libsivelab is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libsivelab is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with libsivelab.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "handleNetworkArgs.h"

using namespace sivelab;

NetworkArgs::NetworkArgs()
  : verbose(false), port(8000), hasHostname(false), hostname("localhost"), 
    hasHostIP(false), hostIP("127.0.0.1"), domain(INET_IPv4)
{
}

void NetworkArgs::process(int argc, char *argv[])
{
  ArgumentParsing argParser;

  argParser.reg("help", "help/usage information", ArgumentParsing::NONE, '?');
  argParser.reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');

  argParser.reg("type", "protocol family AF_INET | AF_INET6", ArgumentParsing::STRING, 't');
  argParser.reg("port", "port number", ArgumentParsing::INT, 'p');
  argParser.reg("hostname", "hostname of machine in string, user-friendly notation", ArgumentParsing::STRING, 'n');
  argParser.reg("hostip", "hostname in dotted-decimal or dotted-quad notation", ArgumentParsing::STRING, 'i');
  argParser.reg("useIPv4", "use IPv4 networking", ArgumentParsing::NONE, '4');
  argParser.reg("useIPv6", "use IPv6 networking", ArgumentParsing::NONE, '6');

  argParser.processCommandLineArgs(argc, argv);

  if (argParser.isSet("help"))
    {
      argParser.printUsage();
      exit(EXIT_SUCCESS);
    }

  verbose = argParser.isSet("verbose");

  argParser.isSet("port", port);

  if (argParser.isSet("hostname", hostname))
    hasHostname = true;

  if (argParser.isSet("hostip", hostIP))
    hasHostIP = true;

  if (argParser.isSet("useIPv4"))
    domain = INET_IPv4;
  else if (argParser.isSet("useIPv6"))
    domain = INET_IPv6;
}

