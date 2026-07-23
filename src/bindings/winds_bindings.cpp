/****************************************************************************
 * Copyright (c) 2026 University of La Rochelle and TIPEE
 *
 * This file is part of QES / pyQES.
 *
 * GPL-3.0 License
 ****************************************************************************/

/** @file winds_bindings.cpp
 * @brief pybind11 bindings for the QES-Winds runner (module pyQES._winds).
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "runners/run_winds.h"

namespace py = pybind11;

PYBIND11_MODULE(_winds, m)
{
  m.doc() = "QES-Winds native bindings (pyQES).";

  py::class_<pyqes::WindsRunResult>(m, "WindsRunResult")
    .def_readonly("winds_out", &pyqes::WindsRunResult::windsOut)
    .def_readonly("winds_wk", &pyqes::WindsRunResult::windsWk)
    .def_readonly("turb_out", &pyqes::WindsRunResult::turbOut)
    .def("__repr__", [](const pyqes::WindsRunResult &r) {
      return "<WindsRunResult winds_out='" + r.windsOut
             + "' winds_wk='" + r.windsWk
             + "' turb_out='" + r.turbOut + "'>";
    });

  m.def(
    "run_winds",
    [](const std::string &xml_path,
       int solve_type,
       const std::string &out_basename,
       bool visu_output,
       bool wksp_output,
       bool turb_output) {
      py::gil_scoped_release release;
      return pyqes::run_winds(xml_path, solve_type, out_basename, visu_output, wksp_output, turb_output);
    },
    py::arg("xml_path"),
    py::arg("solve_type") = 1,
    py::arg("out_basename") = "qes",
    py::arg("visu_output") = true,
    py::arg("wksp_output") = true,
    py::arg("turb_output") = false,
    "Run the QES-Winds solver on a QES XML file and return the output paths.");
}
