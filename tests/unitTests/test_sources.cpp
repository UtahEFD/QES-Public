//
// Created by Fabien Margairaz on 11/25/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "QESDataTransport.h"

#include "util/ManagedContainer.h"
#include "plume/Particle.h"
#include "plume/TracerParticle.h"


class SourceComponent
{
public:
  SourceComponent(){};
  virtual ~SourceComponent(){};
  virtual void set(QESDataTransport &data) = 0;

protected:
};

class SourceGeometry : public SourceComponent
{
public:
  SourceGeometry(){};
  ~SourceGeometry(){};
  void set(QESDataTransport &data) override
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


class MassGenerator : public SourceComponent
{
public:
  MassGenerator(){};
  ~MassGenerator(){};
  void set(QESDataTransport &data) override
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
  explicit Source(const int &n) : id(n) {}
  ~Source()
  {
    for (auto c : components)
      delete c;
  }
  void addComponent(SourceComponent *c) { components.emplace_back(c); }
  void setComponents()
  {
    for (auto c : components)
      c->set(data);
  }
  void print()
  {
    std::cout << id << std::endl;
  }

  QESDataTransport data;

private:
  Source() : id(-1) {}
  int id;
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
}

TEST_CASE("Source generator")
{
  std::vector<Source *> sources;
  ManagedContainer<TracerParticle> particles(100);

  sources.emplace_back(new Source(0));
  sources.emplace_back(new Source(1));
  for (auto s : sources) {
    s->addComponent(new SourceGeometry());
    s->addComponent(new MassGenerator());
  }

  for (auto s : sources)
    s->setComponents();

  for (auto s : sources)
    setParticle(s, particles);

  for (auto s : sources)
    delete s;
}