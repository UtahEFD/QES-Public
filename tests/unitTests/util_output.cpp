//
// Created by Fabien Margairaz on 4/1/24.
//
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <list>
#include <string>

#include "util/QEStime.h"
#include "QESFileOutput.h"
#include "QESOutput.h"
#include "QESNetCDFOutput.h"

class Concentration : public QESOutput
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

  void setOutputFields() override
  {
    std::cout << "[Concentration] call set output" << std::endl;

    m_output_file->createDimension("x_c", "x-center collection box", "m", &xBoxCen);
    m_output_file->createDimension("y_c", "y-center collection box", "m", &yBoxCen);
    m_output_file->createDimension("z_c", "z-center collection box", "m", &zBoxCen);

    m_output_file->createDimensionSet("concentration", { "t", "z_c", "y_c", "x_c" });

    m_output_file->createField("t_avg", "Averaging time", "s", "time", &ongoingAveragingTime);
    m_output_file->createField("p", "number of particle per box", "#ofPar", "concentration", &pBox);
    m_output_file->createField("c", "concentration", "g m-3", "concentration", &conc);

    output_fields = { "t_avg", "p", "c" };
    // need to have a way to track which variable is in which subject their own variables.
  }

protected:
  // output concentration storage variables
  float ongoingAveragingTime;
  std::vector<float> xBoxCen, yBoxCen, zBoxCen;// list of x,y, and z points for the concentration sampling box information
  std::vector<int> pBox;// sampling box particle counter (for average)
  std::vector<float> conc;// concentration values (for output)
};

class Spectra : public QESOutput
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

  void setOutputFields() override
  {
    std::cout << "[Spectra] call set output" << std::endl;

    m_output_file->createDimension("k_x", "x-wavenumber", "m-1", &xBoxCen);
    m_output_file->createDimension("k_y", "y-wavenumber", "m-1", &yBoxCen);

    m_output_file->createDimensionSet("spectra", { "t", "k_y", "k_x" });

    m_output_file->createField("t_collection", "Collecting time", "s", "time", &ongoingAveragingTime);
    m_output_file->createField("s", "spectra", "m-3", "spectra", &sp);

    // need to have a way to track which variable is in which subject their own variables.
    output_fields = { "t_collection", "s" };
  }

protected:
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
  virtual void attach(QESFileOutput *out, QESOutputInterface *)
  {
    /*std::vector<QESOutputInterface *> output_ptr;
    for (auto p : output_ptr) {
      std::cout << "output loop" << std::endl;
      p->setOutput(out);
    }*/
  }

  virtual void detach(QESFileOutput *out, QESOutputInterface *) = 0;
  virtual void Notify(QESOutputInterface *, const std::string &) = 0;

protected:
  QESOutputDirector() = default;

  std::string basename;
  std::vector<QESOutputInterface *> tmp1;
  std::vector<QESFileOutput *> files;
};

TEST_CASE("unit test of output system")
{
  // QESOutputDirector *testOutput = new QESOutputDirector("test");

  QESFileOutput *testFile = new QESNetCDFOutput("test.nc");

  QESOutput *concentration = new Concentration();
  testFile->attach(concentration);
  // concentration->setOutputFields();

  QESOutput *spectra = new Spectra();
  testFile->attach(spectra);
  // spectra->setOutputFields();


  QEStime t;

  testFile->newTimeEntry(t);

  // concentration->compute()
  // concentration->save(t);
  // spectra->save(t);

  // testFile->save(t);

  t += 120;

  testFile->newTimeEntry(t);

  // concentration->save(t);
  // spectra->save(t);

  // testFile->save(t);


  // QESOutputInterface *spectra = new QESOutput();
  // testFile->attach(spectra);
  // spectra->setOutputFields();
}