//
// Created by Fabien Margairaz on 11/25/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "QESDataTransport.h"

#include "util/ManagedContainer.h"
#include "plume/Particle.h"
#include "plume/Random.h"
#include "plume/TracerParticle.h"


class SourceComponent
{
public:
  SourceComponent(){};
  virtual ~SourceComponent(){};
  virtual void generate(const int &, QESDataTransport &) = 0;

protected:
};

class SourceGeometry : public SourceComponent
{
public:
  SourceGeometry(){};
  ~SourceGeometry(){};
  void generate(const int &n, QESDataTransport &data) override
  {
    data.put("x", getPosition(10));
    data.put("y", getPosition(10));
    data.put("z", getPosition(10));
  }

private:
  std::vector<float> getPosition(int n)
  {
    std::vector<float> tmp(n, 0.0);
    return tmp;
  }
};

class SourceGeometryLine : public SourceComponent
{
public:
  SourceGeometryLine() = default;
  SourceGeometryLine(const float &posX_0, const float &posY_0, const float &posZ_0, const float &posX_1, const float &posY_1, const float &posZ_1)
    : m_posX_0(posX_0), m_posY_0(posY_0), m_posZ_0(posZ_0), m_posX_1(posX_1), m_posY_1(posY_1), m_posZ_1(posZ_1)
  {
    m_diffX = m_posX_1 - m_posX_0;
    m_diffY = m_posY_1 - m_posY_0;
    m_diffZ = m_posZ_1 - m_posZ_0;
  }

  ~SourceGeometryLine(){};
  void generate(const int &n, QESDataTransport &data) override
  {

    std::vector<vec3> init(n);
    Random prng;

    for (int k = 0; k < n; ++k) {
      float t = prng.uniRan();
      init[k] = { m_posX_0 + t * m_diffX, m_posY_0 + t * m_diffY, m_posZ_0 + t * m_diffZ };
      // init[k] = VectorMath::
    }
    data.put("init", init);

    std::vector<float> x, y, z;
    getPosition(n, x);
    data.put("x", x);

    getPosition(n, y);
    data.put("y", y);

    getPosition(n, z);
    data.put("z", z);
  }

private:
  void getPosition(int n, std::vector<float> &v)
  {
    v.resize(n, 0.0);
  }

  // vec3 m_pos_start,m_pos_end,m_diff;
  float m_posX_0, m_posY_0, m_posZ_0;
  float m_posX_1, m_posY_1, m_posZ_1;
  float m_diffX, m_diffY, m_diffZ;
};


class MassGenerator : public SourceComponent
{
public:
  MassGenerator(){};
  ~MassGenerator(){};
  void generate(const int &n, QESDataTransport &data) override
  {
    data.put("mass", getMass());
  }

private:
  float getMass()
  {
    return 0.0;
  }
};

class Source
{
public:
  explicit Source(const int &id) : m_id(id) {}
  ~Source()
  {
    for (auto c : components)
      delete c;
  }
  void addComponent(SourceComponent *c) { components.emplace_back(c); }
  void setComponents()
  {
    // query how many particle need to be released
    // if (currTime >= m_releaseType->m_releaseStartTime && currTime <= m_releaseType->m_releaseEndTime) {
    //  return m_releaseType->m_particlePerTimestep;
    int n = 10;
    for (auto c : components)
      c->generate(n, data);
  }
  void print()
  {
    std::cout << m_id << std::endl;
  }

  QESDataTransport data;

private:
  Source() : m_id(-1) {}
  int m_id;
  std::vector<SourceComponent *> components;
};

// this function will be part of the Tracer Model, making the sources agnostics to the
// particle type
void setParticle(Source *s, ManagedContainer<TracerParticle> &p)
{
  // to do (for new source framework):
  // - query the source for the number of particle to be released
  // - format the particle container and return a list of pointer to the new particles
  // this need to be refined... is there an option to avoid copy?
  auto x = s->data.get<std::vector<float>>("x");
  for (size_t k = 0; k < 10; ++k) {
    p.get(k)->xPos_init = x[k];
    p.get(k)->yPos_init = s->data.get<std::vector<float>>("y")[k];
  }
  p.get(0)->m = s->data.get<float>("mass");
  // here the parameters can be dependent on the type of particles
  // data can be queried for optional components.

  // if cuda is used, we can directly copy to the device here:
  // copy(s->data.get<float>("mass").data())
}

TEST_CASE("Source generator")
{
  std::vector<Source *> sources;
  ManagedContainer<TracerParticle> particles(100);

  sources.emplace_back(new Source(0));
  sources.emplace_back(new Source(1));
  for (auto s : sources) {
    s->addComponent(new SourceGeometryLine(0, 1, 0, 1, 0, 1));
    s->addComponent(new MassGenerator());
  }

  for (auto s : sources)
    s->setComponents();

  for (auto s : sources)
    setParticle(s, particles);

  for (auto s : sources)
    delete s;
}