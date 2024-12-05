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
#include "plume/PI_ReleaseType_instantaneous.hpp"
#include "plume/PI_ReleaseType_continuous.hpp"
#include "plume/PI_ReleaseType_duration.hpp"

class SourceComponent
{
public:
  SourceComponent() = default;
  virtual ~SourceComponent() = default;
  virtual void generate(const QEStime &, const int &, QESDataTransport &) = 0;

protected:
};

class ReleaseController
{
public:
  ReleaseController() = default;
  virtual ~ReleaseController() = default;

  virtual QEStime startTime() = 0;
  virtual QEStime endTime() = 0;
  virtual int nbrParticle(const QEStime &) = 0;
  virtual float massParticle(const QEStime &) = 0;

protected:
};

class ReleaseController_XML : public ReleaseController
{
public:
  explicit ReleaseController_XML(PI_ReleaseType *in)
    : m_startTime(in->m_releaseStartTime), m_endTime(in->m_releaseEndTime),
      m_particlePerTimestep(in->m_particlePerTimestep), m_massPerParticle(in->m_massPerParticle)
  {}
  ~ReleaseController_XML() override = default;

  QEStime startTime() override { return m_startTime; }
  QEStime endTime() override { return m_endTime; }
  int nbrParticle(const QEStime &currTime) override { return m_particlePerTimestep; }
  float massParticle(const QEStime &currTime) override { return m_massPerParticle; }

protected:
  QEStime m_startTime;
  QEStime m_endTime;
  int m_particlePerTimestep{};
  float m_massPerParticle{};

private:
  ReleaseController_XML() = default;
};

class SourceIDGen : public SourceComponent
{
public:
  SourceIDGen()
  {
    id_gen = IDGenerator::getInstance();
  }
  ~SourceIDGen() override = default;
  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override
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
  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override
  {
    data.put("position", std::vector<vec3>(n, pos));
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

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override
  {

    std::vector<vec3> init(n);
    Random prng;

    for (int k = 0; k < n; ++k) {
      float t = prng.uniRan();
      // init[k] = { m_posX_0 + t * m_diffX, m_posY_0 + t * m_diffY, m_posZ_0 + t * m_diffZ };
      init[k] = VectorMath::add(m_pos_0, VectorMath::multiply(t, m_diff));
    }
    data.put("position", init);
  }

private:
  SourceGeometryLine() = default;

  // vec3 m_pos_start,m_pos_end,m_diff;
  vec3 m_pos_0{}, m_pos_1{}, m_diff{};
};


class SetMass : public SourceComponent
{
public:
  explicit SetMass(ReleaseController *in)
    : m_release(in)
  {}
  ~SetMass() override = default;

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override
  {
    data.put("mass", std::vector<float>(n, m_release->massParticle(currTime)));
  }

private:
  SetMass() = default;

  ReleaseController *m_release{};
};

class SetPhysicalProperties : public SourceComponent
{
public:
  explicit SetPhysicalProperties(float particleDiameter)
    : m_particleDiameter(particleDiameter)
  {}
  ~SetPhysicalProperties() override = default;

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override
  {
    data.put("diameter", std::vector<float>(n, m_particleDiameter));
    data.put("density", std::vector<float>(n, m_particleDensity));
  }

private:
  SetPhysicalProperties() = default;

  float m_particleDiameter{};
  float m_particleDensity{};
};

class Source
{
public:
  Source(const int &id, ReleaseController *release)
    : m_id(id), m_release(release)
  {
    m_components.emplace_back(new SourceIDGen());
    m_components.emplace_back(new SetMass(release));
  }
  virtual ~Source()
  {
    for (auto c : m_components)
      delete c;
  }

  virtual bool isActive(const QEStime &currTime) const
  {
    return (currTime >= m_release->startTime() && currTime <= m_release->endTime());
  }

  void addComponent(SourceComponent *c)
  {
    m_components.emplace_back(c);
  }

  int getID() const
  {
    return m_id;
  }

  virtual int generate(const QEStime &currTime)
  {
    // query how many particle need to be released
    if (isActive(currTime)) {
      // m_releaseType->m_particlePerTimestep;
      int n = m_release->nbrParticle(currTime);
      for (auto c : m_components)
        c->generate(currTime, n, data);

      // update source counter
      if (data.contains("mass")) {
        for (auto m : data.get_ref<std::vector<float>>("mass"))
          total_mass += m;
      }
      total_particle_released += n;

      return n;
    } else {
      data.clear();
      return 0;
    }
  }
  void print() const
  {
    std::cout << m_id << std::endl;
  }

  QESDataTransport data;

protected:
  explicit Source(const int &id) : m_id(id) {}

private:
  Source() : m_id(-1) {}
  int m_id;

  std::vector<SourceComponent *> m_components{};

  ReleaseController *m_release{};

  float total_mass = 0;
  int total_particle_released = 0;
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
    p.last_added()->pos_init = s->data.get_ref<std::vector<vec3>>("position")[k];
    p.last_added()->m = s->data.get_ref<std::vector<float>>("mass")[k];
    p.last_added()->d = s->data.get_ref<std::vector<float>>("diameter")[k];
  }

  // here the parameters can be dependent on the type of particles
  // data can be queried for optional components.

  // if cuda is used, we can directly copy to the device here:
  // copy(s->data.get_ref<float>("mass").data())
}

TEST_CASE("Source generator")
{
  std::vector<Source *> sources;
  ManagedContainer<TracerParticle> particles(50);

  QEStime time("2020-01-01T00:00");
  // sources.emplace_back(new Source(0, time, time + 10));
  PI_ReleaseType *pt_dur = new PI_ReleaseType_duration();
  sources.emplace_back(new Source(0, new ReleaseController_XML(pt_dur)));
  sources.back()->addComponent(new SourceGeometryLine({ 0, 0, 0 }, { 1, 1, 1 }));
  sources.back()->addComponent(new SetPhysicalProperties(0.0));

  // auto *source_tmp = new Source(1, time, time + 10);
  PI_ReleaseType *pt_cont = new PI_ReleaseType_continuous();
  auto *source_tmp = new Source(1, new ReleaseController_XML(pt_cont));
  source_tmp->addComponent(new SourceGeometryPoint({ 1, 1, 1 }));
  source_tmp->addComponent(new SetPhysicalProperties(0.001));
  sources.push_back(source_tmp);


  int nbr_new_particle = 0;
  for (auto s : sources)
    nbr_new_particle += s->generate(time);

  // particles.sweep(nbr_new_particle);

  for (auto s : sources) {
    if (s->isActive(time))
      setParticle(time, s, particles);
  }

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