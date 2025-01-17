//
// Created by Fabien Margairaz on 11/25/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "util/QESDataTransport.h"


#include "util/QEStime.h"

#include "plume/PLUMEGeneralData.h"
#include "plume/PLUMEInputData.h"
#include "plume/SourceIDGen.h"

#include "plume/ManagedContainer.h"
#include "plume/Particle.h"
#include "plume/Random.h"
#include "plume/TracerParticle.h"
#include "plume/ParticleIDGen.h"

#include "plume/SourceGeometryPoint.h"
#include "plume/SourceReleaseController.h"
#include "plume/PI_Source.hpp"
#include "plume/PI_SourceComponent.hpp"
#include "plume/PI_SourceGeometry_Cube.hpp"
#include "plume/PI_SourceGeometry_FullDomain.hpp"
#include "plume/PI_SourceGeometry_Line.hpp"
#include "plume/PI_SourceGeometry_Point.hpp"
#include "plume/PI_SourceGeometry_SphereShell.hpp"

#include "plume/PI_ReleaseType.hpp"
#include "plume/PI_ReleaseType_instantaneous.hpp"
#include "plume/PI_ReleaseType_continuous.hpp"
#include "plume/PI_ReleaseType_duration.hpp"

/*class SourceComponent
{
public:
  SourceComponent() = default;
  virtual ~SourceComponent() = default;
  virtual void generate(const QEStime &, const int &, QESDataTransport &) = 0;

protected:
};*/
/*
class ReleaseController
{
public:
  ReleaseController() = default;
  virtual ~ReleaseController() = default;

  virtual QEStime startTime() = 0;
  virtual QEStime endTime() = 0;
  virtual int particles(const QEStime &) = 0;
  virtual float mass(const QEStime &) = 0;

protected:
};

class ReleaseController_base : public ReleaseController
{
public:
  ReleaseController_base(const QEStime &s_time, const QEStime &e_time, const int &nbr_part, const float &total_mass)
    : m_startTime(s_time), m_endTime(e_time),
      m_particlePerTimestep(nbr_part), m_massPerTimestep(total_mass)
  {}
  ~ReleaseController_base() override = default;

  QEStime startTime() override { return m_startTime; }
  QEStime endTime() override { return m_endTime; }
  int particles(const QEStime &currTime) override { return m_particlePerTimestep; }
  float mass(const QEStime &currTime) override { return m_massPerTimestep; }

protected:
  QEStime m_startTime;
  QEStime m_endTime;
  int m_particlePerTimestep{};
  float m_massPerTimestep{};

private:
  ReleaseController_base() = default;
};
*/


/*class SourceGeometryPoint : public SourceComponent
{
public:
  explicit SourceGeometryPoint(const vec3 &x)
    : m_pos(x)
  {}
  explicit SourceGeometryPoint(const PI_SourceGeometry_Point *param)
    : m_pos({ (float)param->posX, (float)param->posY, (float)param->posZ })
  {}
  ~SourceGeometryPoint() override = default;
  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override
  {
    data.put("position", std::vector<vec3>(n, m_pos));
  }

private:
  SourceGeometryPoint() = default;

  vec3 m_pos{};
};

class SourceGeometryLine : public SourceComponent
{
public:
  SourceGeometryLine(const vec3 &pos_0, const vec3 &pos_1)
    : m_pos_0(pos_0), m_pos_1(pos_1)
  {
    m_diff = VectorMath::subtract(m_pos_1, m_pos_0);
  }
  explicit SourceGeometryLine(const PI_SourceGeometry_Line *param)
    : m_pos_0({ (float)param->posX_0, (float)param->posY_0, (float)param->posZ_0 }),
      m_pos_1({ (float)param->posX_1, (float)param->posY_1, (float)param->posZ_1 })
  {}

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

class SourceGeometryFullDomain : public SourceComponent
{
public:
  SourceGeometryFullDomain(const PLUMEGeneralData *PGD)
  {
    prng = std::mt19937(rd());// Standard mersenne_twister_engine seeded with rd()
    uniform = std::uniform_real_distribution<float>(0.0, 1.0);

    PGD->interp->getDomainBounds(xDomainStart, yDomainStart, zDomainStart, xDomainEnd, yDomainEnd, zDomainEnd);
  }

  ~SourceGeometryFullDomain() override = default;

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override
  {
    std::vector<vec3> init(n);

    for (int k = 0; k < n; ++k) {
      // init[k] = { m_posX_0 + t * m_diffX, m_posY_0 + t * m_diffY, m_posZ_0 + t * m_diffZ };
      init[k]._1 = uniform(prng) * (xDomainEnd - xDomainStart) + xDomainStart;
      init[k]._2 = uniform(prng) * (yDomainEnd - yDomainStart) + yDomainStart;
      init[k]._3 = uniform(prng) * (zDomainEnd - zDomainStart) + zDomainStart;
    }
    data.put("position", init);
  }

private:
  SourceGeometryFullDomain() = default;

  float xDomainStart = -1.0;
  float yDomainStart = -1.0;
  float zDomainStart = -1.0;
  float xDomainEnd = -1.0;
  float yDomainEnd = -1.0;
  float zDomainEnd = -1.0;

  std::random_device rd;// Will be used to obtain a seed for the random number engine
  std::mt19937 prng;// Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> uniform;
};

class SourceGeometrySphereShell : public SourceComponent
{
public:
  SourceGeometrySphereShell(const vec3 &min, const float &radius)
    : m_x(min), m_radius(radius)
  {
    prng = std::mt19937(rd());// Standard mersenne_twister_engine seeded with rd()
    normal = std::normal_distribution<float>(0.0, 1.0);
  }
  explicit SourceGeometrySphereShell(const PI_SourceGeometry_SphereShell *param)
    : m_x({ param->posX, param->posY, param->posZ }),
      m_radius(param->radius)
  {
    prng = std::mt19937(rd());// Standard mersenne_twister_engine seeded with rd()
    normal = std::normal_distribution<float>(0.0, 1.0);
  }

  ~SourceGeometrySphereShell() override = default;

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override
  {
    std::vector<vec3> init(n);

    for (int k = 0; k < n; ++k) {
      // init[k] = { m_posX_0 + t * m_diffX, m_posY_0 + t * m_diffY, m_posZ_0 + t * m_diffZ };
      float nx = normal(prng);
      float ny = normal(prng);
      float nz = normal(prng);
      float overn = 1 / sqrt(nx * nx + ny * ny + nz * nz);
      init[k]._1 = m_x._1 + m_radius * nx * overn;
      init[k]._2 = m_x._2 + m_radius * ny * overn;
      init[k]._3 = m_x._3 + m_radius * nz * overn;
    }
    data.put("position", init);
  }

private:
  SourceGeometrySphereShell() = default;

  vec3 m_x{};
  float m_radius = 0;

  std::random_device rd;// Will be used to obtain a seed for the random number engine
  std::mt19937 prng;// Standard mersenne_twister_engine seeded with rd()
  std::normal_distribution<float> normal;
};

class SourceGeometryCube : public SourceComponent
{
public:
  SourceGeometryCube(const vec3 &min, const vec3 &max)
    : m_min(min), m_max(max)
  {
    prng = std::mt19937(rd());// Standard mersenne_twister_engine seeded with rd()
    uniform = std::uniform_real_distribution<float>(0.0, 1.0);
  }
  explicit SourceGeometryCube(const PI_SourceGeometry_Cube *param)
    : m_min({ (float)param->m_minX, (float)param->m_minY, (float)param->m_minZ }),
      m_max({ (float)param->m_maxX, (float)param->m_maxY, (float)param->m_maxZ })
  {
    prng = std::mt19937(rd());// Standard mersenne_twister_engine seeded with rd()
    uniform = std::uniform_real_distribution<float>(0.0, 1.0);
  }

  ~SourceGeometryCube() override = default;

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override
  {
    std::vector<vec3> init(n);

    for (int k = 0; k < n; ++k) {
      // init[k] = { m_posX_0 + t * m_diffX, m_posY_0 + t * m_diffY, m_posZ_0 + t * m_diffZ };
      init[k]._1 = uniform(prng) * (m_max._1 - m_min._1) + m_min._1;
      init[k]._2 = uniform(prng) * (m_max._2 - m_min._2) + m_min._2;
      init[k]._3 = uniform(prng) * (m_max._3 - m_min._3) + m_min._3;
    }
    data.put("position", init);
  }

private:
  SourceGeometryCube() = default;

  vec3 m_min{};
  vec3 m_max{};

  std::random_device rd;// Will be used to obtain a seed for the random number engine
  std::mt19937 prng;// Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> uniform;
};*/
/*
class SetMass : public SourceComponent
{
public:
  explicit SetMass(SourceReleaseController *in)
    : m_release(in)
  {}
  ~SetMass() override = default;

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override
  {
    data.put("mass", std::vector<float>(n, m_release->mass(currTime) / (float)n));
  }

private:
  SetMass() = default;

  SourceReleaseController *m_release{};
};
*/

/*class SetPhysicalProperties : public SourceComponent
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
};*/

/*class Source_test
{
public:
  Source_test()
  {
    auto sourceID = SourceIDGen::getInstance();
    m_id = sourceID->get();

    m_components.emplace_back(new SetParticleID());
  }

  Source_test(SourceReleaseController *release)
    : m_release(release)
  {
    auto sourceID = SourceIDGen::getInstance();
    m_id = sourceID->get();

    m_components.emplace_back(new SetParticleID());
    m_components.emplace_back(new SetMass(release));
  }
  virtual ~Source_test()
  {
    delete m_release;
    for (auto c : m_components)
      delete c;
  }

  virtual bool isActive(const QEStime &currTime) const
  {
    return (currTime >= m_release->startTime() && currTime <= m_release->endTime());
  }

  int getID() const { return m_id; }

  void addRelease(SourceReleaseController *r) { m_release = r; }
  void addComponent(SourceComponent *c) { m_components.emplace_back(c); }


  virtual int generate(const QEStime &currTime)
  {
    // query how many particle need to be released
    if (isActive(currTime)) {
      // m_releaseType->m_particlePerTimestep;
      int n = m_release->particles(currTime);
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
private:
  int m_id;

  std::vector<SourceComponent *> m_components{};

  SourceReleaseController *m_release{};

  float total_mass = 0;
  int total_particle_released = 0;
};*/

/*
class MySourceBuilder : public SourceComponentBuilderInterface
{
private:
  //....
public:
  MySourceBuilder(){};

  SourceComponent *create(QESDataTransport &data) override
  {
    //....
  }
};


class SourceBuilder
{
public:
  SourceBuilder()
  {
    source = new Source_test();
  }
  Source_test *returnSource() { return source; }

protected:
  Source_test *source;
};

class SourceBuilder_XML : public SourceBuilder
{
public:
  SourceBuilder_XML(PI_Source *pi_s) : SourceBuilder()
  {
    QESDataTransport data;
    source->addComponent(new SetParticleID());
    source->addRelease(pi_s->m_releaseType->create(data));
    source->addComponent(pi_s->m_sourceGeometry->create(data));
  }
};
*/


// this function will be part of the Tracer Model, making the sources agnostics to the
// particle type
void setParticle(const QEStime &currTime, Source *s, ManagedContainer<TracerParticle> &p)
{
  // to do (for new source framework):
  // - query the source for the number of particle to be released
  // - format the particle container and return a list of pointer to the new particles
  // this need to be refined... is there an option to avoid copy?
  for (size_t k = 0; k < s->data().get_ref<std::vector<u_int32_t>>("ID").size(); ++k) {
    // p.get(k)->pos_init = x[k];
    p.insert();
    p.last_added()->ID = s->data().get_ref<std::vector<u_int32_t>>("ID")[k];
    p.last_added()->sourceIdx = s->getID();
    p.last_added()->timeStrt = currTime;
    p.last_added()->pos_init = s->data().get_ref<std::vector<vec3>>("position")[k];

    if (s->data().contains("mass")) {
      p.last_added()->m = s->data().get_ref<std::vector<float>>("mass")[k];
    }
    // p.last_added()->d = s->data.get_ref<std::vector<float>>("diameter")[k];
    // p.last_added()->rho = s->data.get_ref<std::vector<float>>("density")[k];
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

  std::string qesPlumeParamFile = QES_DIR;
  qesPlumeParamFile.append("/tests/unitTests/plume_input_parameters.xml");
  auto PID = new PlumeInputData(qesPlumeParamFile);

  /*PI_ReleaseType *pt_dur = new PI_ReleaseType_duration();
  sources.emplace_back(new Source_test(pt_dur->create(), {}));
  sources.back()->addComponent(new SourceGeometryLine({ 0, 0, 0 }, { 1, 1, 1 }));
  sources.back()->addComponent(new SetPhysicalProperties(0.0));
   */

  // auto *source_tmp = new Source(1, time, time + 10);
  /*PI_ReleaseType *pt_cont = new PI_ReleaseType_continuous();
  auto *source_tmp = new Source_test(pt_cont->create(), {});
  source_tmp->addComponent(new SourceGeometryPoint({ 1, 1, 1 }));
  source_tmp->addComponent(new SetPhysicalProperties(0.001));
  sources.push_back(source_tmp);*/

  // sources.emplace_back(new Source(new SourceReleaseController_base(time, time + 10, 10, 0.1)));
  // sources.back()->addComponent(new SourceGeometryLine({ 0, 0, 0 }, { 1, 1, 1 }));
  //  sources.back()->addComponent(new SetPhysicalProperties(0.0));


  /*PI_SourceComponent *ptr_geom = new PI_SourceGeometry_Point();
  sources.emplace_back(new Source_test(new SourceReleaseController_base(time, time + 10, 10, 0.1),
                                       { ptr_geom->create(),
                                         new SetPhysicalProperties(0.0) }));
  */

  QESDataTransport data;
  for (auto p : PID->particleParams->particles) {
    for (auto s : p->sources) {
      sources.push_back(s->create(data));
      // SourceBuilder_XML sourceBuilder(s);
      // sources.push_back(sourceBuilder.returnSource());
    }
  }

  int nbr_new_particle = 0;
  for (auto s : sources)
    nbr_new_particle += s->generate(time);

  std::cout << "NEW PARTICLES " << nbr_new_particle << std::endl;
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

  for (auto p = particles.begin(); p != particles.end(); p++)
    std::cout << p->pos_init._1 << " ";
  std::cout << std::endl;


  for (auto s : sources)
    delete s;
}