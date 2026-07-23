/****************************************************************************
 * Copyright (c) 2026 University of La Rochelle and TIPEE
 *
 * This file is part of QES / pyQES.
 *
 * GPL-3.0 License
 ****************************************************************************/

/** @file fire_bindings.cpp
 * @brief pybind11 bindings for the QES-Fire runner (module pyQES._fire).
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "runners/run_fire.h"

namespace py = pybind11;

PYBIND11_MODULE(_fire, m)
{
  m.doc() = "QES-Fire native bindings (pyQES).";

  py::class_<pyqes::FireRunResult>(m, "FireRunResult")
    .def_readonly("fire_out", &pyqes::FireRunResult::fireOut)
    .def_readonly("plume_out", &pyqes::FireRunResult::plumeOut)
    .def("__repr__", [](const pyqes::FireRunResult &r) {
      return "<FireRunResult fire_out='" + r.fireOut
             + "' plume_out='" + r.plumeOut + "'>";
    });

  m.def(
    "run_fire",
    [](const std::string &winds_xml_path,
       const std::string &plume_xml_path,
       int solve_type,
       const std::string &out_basename,
       bool comp_turb,
       bool fire_winds_off) {
      py::gil_scoped_release release;
      return pyqes::run_fire(winds_xml_path, plume_xml_path, solve_type, out_basename, comp_turb, fire_winds_off);
    },
    py::arg("winds_xml_path"),
    py::arg("plume_xml_path") = "",
    py::arg("solve_type") = 1,
    py::arg("out_basename") = "qes",
    py::arg("comp_turb") = false,
    py::arg("fire_winds_off") = false,
    "Run the coupled QES-Fire workflow and return the output paths.");
}
