// src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "quantize.h"

namespace py = pybind11;

PYBIND11_MODULE(mymodule, m) {
    m.def("quantize_tensor", &quantize_tensor, "A function that quantizes a PyTorch tensor",
          py::arg("tensor"), py::arg("scale"));
}
