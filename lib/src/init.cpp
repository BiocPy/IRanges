#include "pybind11/pybind11.h"

namespace py = pybind11;

void init_coverage(pybind11::module &);
void init_interranges(pybind11::module &);
void init_nclist(pybind11::module &);
void init_nclistsearch(pybind11::module &);

PYBIND11_MODULE(lib_iranges, m) {
    m.doc() = "Iranges cpp implementations";

    init_coverage(m);
    init_interranges(m);
    init_nclist(m);
    init_nclistsearch(m);
}
