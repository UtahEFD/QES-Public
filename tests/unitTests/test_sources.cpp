//
// Created by Fabien Margairaz on 11/25/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "QESDataTransport.h"

#include "util/QEStime.h"

#include "plume/ManagedContainer.h"
#include "plume/Particle.h"
#include "plume/Random.h"
#include "plume/TracerParticle.h"
#include "plume/IDGenerator.h"

#include "plume/PI_ReleaseType.hpp"


class SourceComponent
{
public:
  SourceComponent() = default;
  virtual ~SourceComponent() = default;
  virtual void generate(const int &, QESDataTransport &) = 0;

protected:
};

class SourceIDGen : public SourceComponent
{
public:
  SourceIDGen()
  {
    id_gen = IDGenerator::getInstance();
  }
  ~SourceIDGen() override = default;
  void generate(const int &n, QESDataTransport &data) override
  {
    std::vector<uint32_t> ids(n);
    id_gen->get(ids);
    data.put("ID", ids);
  }

private:
  IDGenerator *id_gen = nullptr;
};

class SourceGeometryPoint : public SourceComponent
{
public:
  SourceGeometryPoint(const vec3 &x) : pos(x)
  {}
  ~SourceGeometryPoint() override = default;
  void generate(const int &n, QESDataTransport &data) override
  {
    data.put("pos", std::vector<vec3>(n, pos));
  }

private:
  SourceGeometryPoint() = default;

  vec3 pos{};
};

class SourceGeometryLine : public SourceComponent
{
public:
  SourceGeometryLine(const vec3 &pos_0, const vec3 &pos_1)
    : m_pos_0(pos_0), m_pos_1(pos_1)
  {
    m_diff = VectorMath::subtract(m_pos_1, m_pos_0);
  }
  ~SourceGeometryLine() override = default;

  void generate(const int &n, QESDataTransport &data) override
  {

    std::vector<vec3> init(n);
    Random prng;

    for (int k = 0; k < n; ++k) {
      float t = prng.uniRan();
      // init[k] = { m_posX_0 + t * m_diffX, m_posY_0 + t * m_diffY, m_posZ_0 + t * m_diffZ };
      init[k] = VectorMath::add(m_pos_0, VectorMath::multiply(t, m_diff));
    }
    data.put("pos", init);
  }

private:
  SourceGeometryLine() = default;

  // vec3 m_pos_start,m_pos_end,m_diff;
  vec3 m_pos_0{}, m_pos_1{}, m_diff{};
};


class MassGeneratorConstant : public SourceComponent
{
public:
  explicit MassGeneratorConstant(float massPerParticle) : m_massPerParticle(massPerParticle)
  {}
  ~MassGeneratorConstant() override = default;

  void generate(const int &n, QESDataTransport &data) override
  {
    data.put("mass", std::vector<float>(n, m_massPerParticle));
  }

private:
  MassGeneratorConstant() = default;

  float m_massPerParticle{};
};

class Source
{
public:
  explicit Source(const int &id, const QEStime &releaseTimeStart, const QEStime &releaseTimeEnd)
    : m_id(id),
      m_releaseStartTime(releaseTimeStart), m_releaseEndTime(releaseTimeEnd)
  {
    components.emplace_back(new SourceIDGen());
  }
  ~Source()
  {
    for (auto c : components)
      delete c;
  }
  void addComponent(SourceComponent *c) { components.emplace_back(c); }
  int getID() { return m_id; }
  int generate(const QEStime &currTime)
  {
    // query how many particle need to be released
    if (currTime >= m_releaseStartTime && currTime <= m_releaseEndTime) {
      // m_releaseType->m_particlePerTimestep;
      int n = 10;
      for (auto c : components)
        c->generate(n, data);

      return n;
    } else {
      return 0;
    }
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

  QEStime m_releaseStartTime;
  QEStime m_releaseEndTime;

  PI_ReleaseType *m_release{};
};

// this function will be part of the Tracer Model, making the sources agnostics to the
// particle type
void setParticle(const QEStime &currTime, Source *s, ManagedContainer<TracerParticle> &p)
{
  // to do (for new source framework):
  // - query the source for the number of particle to be released
  // - format the particle container and return a list of pointer to the new particles
  // this need to be refined... is there an option to avoid copy?
  for (size_t k = 0; k < 10; ++k) {
    // p.get(k)->pos_init = x[k];
    p.insert();
    p.last_added()->ID = s->data.get_ref<std::vector<u_int32_t>>("ID")[k];
    p.last_added()->sourceIdx = s->getID();
    p.last_added()->timeStrt = currTime;
    p.last_added()->pos_init = s->data.get_ref<std::vector<vec3>>("pos")[k];
    p.last_added()->m = s->data.get_ref<std::vector<float>>("mass")[k];
  }

  // here the parameters can be dependent on the type of particles
  // data can be queried for optional components.

  // if cuda is used, we can directly copy to the device here:
  // copy(s->data.get<float>("mass").data())
}

TEST_CASE("Source generator")
{
  std::vector<Source *> sources;
  ManagedContainer<TracerParticle> particles(50);

  QEStime time("2020-01-01T00:00");
  sources.emplace_back(new Source(0, time, time + 10));
  sources.back()->addComponent(new SourceGeometryLine({ 0, 0, 0 }, { 1, 1, 1 }));
  sources.back()->addComponent(new MassGeneratorConstant(1));

  auto *source_tmp = new Source(1, time, time + 10);
  source_tmp->addComponent(new SourceGeometryPoint({ 1, 1, 1 }));
  source_tmp->addComponent(new MassGeneratorConstant(0.001));
  sources.push_back(source_tmp);


  int nbr_new_particle = 0;
  for (auto s : sources)
    nbr_new_particle += s->generate(time);

  // particles.sweep(nbr_new_particle);

  for (auto s : sources)
    setParticle(time, s, particles);


  for (auto p = particles.begin(); p != particles.end(); p++)
    std::cout << p->ID << " ";
  std::cout << std::endl;

  for (auto p = particles.begin(); p != particles.end(); p++)
    std::cout << p->sourceIdx << " ";
  std::cout << std::endl;

  for (auto p = particles.begin(); p != particles.end(); p++)
    std::cout << p->m << " ";
  std::cout << std::endl;

  for (auto s : sources)
    delete s;
}