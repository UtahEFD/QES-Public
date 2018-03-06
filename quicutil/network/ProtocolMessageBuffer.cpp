#include <cstring>

#include "ProtocolMessageBuffer.h"

using namespace sivelab;

ProtocolMessageBuffer::ProtocolMessageBuffer()
  : m_msgBuffer(0), m_msgSz(0)
{
}

ProtocolMessageBuffer::~ProtocolMessageBuffer()
{
  if (m_msgBuffer) delete [] m_msgBuffer;
}

void ProtocolMessageBuffer::addBufferData(const char *buf, int sz)
{
  if (m_msgBuffer)
    {
      char *newBuffer = new char[sz + m_msgSz];
      memcpy(newBuffer, m_msgBuffer, m_msgSz);
      memcpy(newBuffer + m_msgSz, buf, sz);
      
      char *tmpPtr = m_msgBuffer;

      m_msgBuffer = newBuffer;
      m_msgSz += sz;

      // clean up old buffer now that its been copied and replaced
      delete [] tmpPtr;
    }
  else
    {
      m_msgSz = sz;
      m_msgBuffer = new char[m_msgSz];
      memcpy(m_msgBuffer, buf, m_msgSz);
    }
}

ostream &operator<<(ostream& os, const ProtocolMessageBuffer& b)
{
  os << b.toString();
  return os;
}
