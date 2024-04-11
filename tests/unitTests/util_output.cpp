//
// Created by Fabien Margairaz on 4/1/24.
//
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <list>
#include <string>

#include "util/QEStime.h"
#include "QESFileOutput.h"
#include "DataSource.h"
#include "QESNetCDFOutput.h"

class Concentration : public DataSource
{
public:
  Concentration()
  {
    int nBoxesX = 5;
    int nBoxesY = 5;
    int nBoxesZ = 2;


    // output concentration storage variables
    xBoxCen.resize(nBoxesX);
    yBoxCen.resize(nBoxesY);
    zBoxCen.resize(nBoxesZ);

    int zR = 0, yR = 0, xR = 0;
    for (int k = 0; k < nBoxesZ; ++k) {
      zBoxCen.at(k) = 0 + (zR * 1) + (1 / 2.0);
      zR++;
    }
    for (int j = 0; j < nBoxesY; ++j) {
      yBoxCen.at(j) = 0 + (yR * 1) + (1 / 2.0);
      yR++;
    }
    for (int i = 0; i < nBoxesX; ++i) {
      xBoxCen.at(i) = 0 + (xR * 1) + (1 / 2.0);
      xR++;
    }

    ongoingAveragingTime = 1000;

    // initialization of the container
    pBox.resize(nBoxesX * nBoxesY * nBoxesZ, 0);
    conc.resize(nBoxesX * nBoxesY * nBoxesZ, 0.0);
  }

  void prepData(QEStime t) override
  {
    save(t);
  }

protected:
  void setOutputFields() override
  {
    std::cout << "[Concentration] call set output" << std::endl;

    defineDimension("x_c", "x-center collection box", "m", &xBoxCen);
    defineDimension("y_c", "y-center collection box", "m", &yBoxCen);
    defineDimension("z_c", "z-center collection box", "m", &zBoxCen);

    defineDimensionSet("concentration", { "t", "z_c", "y_c", "x_c" });

    defineVariable("t_avg", "Averaging time", "s", "time", &ongoingAveragingTime);
    defineVariable("p", "number of particle per box", "#ofPar", "concentration", &pBox);
    defineVariable("c", "concentration", "g m-3", "concentration", &conc);

    //  need to have a way to track which variable is in which subject their own variables.
    // m_output_fields = { "t_avg", "p", "c" };
  }

  // output concentration storage variables
  float ongoingAveragingTime;
  std::vector<float> xBoxCen, yBoxCen, zBoxCen;// list of x,y, and z points for the concentration sampling box information
  std::vector<int> pBox;// sampling box particle counter (for average)
  std::vector<float> conc;// concentration values (for output)
};

class Spectra : public DataSource
{
public:
  Spectra()
  {
    int nBoxesX = 5;
    int nBoxesY = 5;

    // output concentration storage variables
    xBoxCen.resize(nBoxesX);
    yBoxCen.resize(nBoxesY);

    int zR = 0, yR = 0, xR = 0;
    for (int j = 0; j < nBoxesY; ++j) {
      yBoxCen.at(j) = 0 + (yR * 1) + (1 / 2.0);
      yR++;
    }
    for (int i = 0; i < nBoxesX; ++i) {
      xBoxCen.at(i) = 0 + (xR * 1) + (1 / 2.0);
      xR++;
    }

    ongoingAveragingTime = 1000;

    // initialization of the container
    sp.resize(nBoxesX * nBoxesY, 0.0);
  }

  void prepData(QEStime t) override
  {
    save(t);
  }

protected:
  void setOutputFields() override
  {
    std::cout << "[Spectra] call set output" << std::endl;

    defineDimension("k_x", "x-wavenumber", "m-1", &xBoxCen);
    defineDimension("k_y", "y-wavenumber", "m-1", &yBoxCen);

    defineDimensionSet("spectra", { "t", "k_y", "k_x" });

    defineVariable("t_collection", "Collecting time", "s", "time", &ongoingAveragingTime);
    defineVariable("s", "spectra", "m-3", "spectra", &sp);

    // need to have a way to track which variable is in which subject their own variables.
    // m_output_fields = { "t_collection", "s" };
  }

  // output concentration storage variables
  float ongoingAveragingTime;
  std::vector<float> xBoxCen, yBoxCen, zBoxCen;// list of x,y, and z points for the concentration sampling box information
  std::vector<float> sp;// concentration values (for output)
};

class QESOutputDirector
{
  /*
   * this class is the observer/mediator interface for QES-output files
   */
public:
  QESOutputDirector(std::string name) : basename(std::move(name))
  {
  }
  ~QESOutputDirector() = default;

  virtual void save(const QEStime &) {}
  virtual void attach(QESFileOutput *out, DataSourceInterface *)
  {
    /*std::vector<QESOutputInterface *> output_ptr;
    for (auto p : output_ptr) {
      std::cout << "output loop" << std::endl;
      p->setOutput(out);
    }*/
  }

  virtual void detach(QESFileOutput *out, DataSourceInterface *) = 0;
  virtual void Notify(DataSourceInterface *, const std::string &) = 0;

protected:
  QESOutputDirector() = default;

  std::string basename;
  std::vector<DataSourceInterface *> tmp1;
  std::vector<QESFileOutput *> files;
};

TEST_CASE("unit test of output system")
{
  // QESOutputDirector *testOutput = new QESOutputDirector("test");

  QESFileOutput *testFile = new QESNetCDFOutput("test.nc");

  DataSource *concentration = new Concentration();
  testFile->attach(concentration);
  // concentration->setOutputFields();

  DataSource *spectra = new Spectra();
  testFile->attach(spectra);
  // spectra->setOutputFields();

  QEStime t;
  testFile->setStartTime(t);

  t += 120;

  testFile->newTimeEntry(t);

  // concentration->compute()
  concentration->prepData(t);
  spectra->prepData(t);

  // testFile->save(t);

  t += 120;

  testFile->newTimeEntry(t);

  // concentration->save(t);
  // spectra->save(t);

  testFile->save(t);


  // QESOutputInterface *spectra = new QESOutput();
  // testFile->attach(spectra);
  // spectra->setOutputFields();
}