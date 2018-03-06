/*
 *  ProtocolMessage.h
 *  NETCODE
 *
 *  Created by Pete Willemsen on 10/03/12.
 *  Copyright 2012 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 */

#ifndef __SIVELAB_PROTOCOL_MESSAGE_BUFFER_H__
#define __SIVELAB_PROTOCOL_MESSAGE_BUFFER_H__

#include <cstdio>
#include <sstream>

#include "util/logstream.h"

namespace sivelab {

  class ProtocolMessageBuffer
  {
  public:
    ProtocolMessageBuffer();
    virtual ~ProtocolMessageBuffer();

    virtual void addBufferData(const char *buf, int sz);

    int length() const { return m_msgSz; }
    const char *byteArray() const { return m_msgBuffer; }

    std::string toString() const { return m_msgBuffer; }

  protected:
    char *m_msgBuffer;
    int m_msgSz;
    
  private:
    friend std::ostream& operator<<(std::ostream& os, const ProtocolMessageBuffer &b);
  };
  
}

#endif
