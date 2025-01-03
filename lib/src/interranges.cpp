#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <tuple>

namespace py = pybind11;

static std::vector<int> get_order(
    const py::array_t<int32>& starts,
    const py::array_t<int32>& widths
) {
    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    int n = starts_r.shape(0);
    
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) {
        order[i] = i;
    }
    
    // Sort by start position, then by width
    std::sort(order.begin(), order.end(),
        [&starts_r, &widths_r](int32 i, int32 j) {
            if (starts_r(i) != starts_r(j)) {
                return starts_r(i) < starts_r(j);
            }
            return widths_r(i) < widths_r(j);
        });
    
    return order;
}

void init_interranges(pybind11::module& m) {
    m.def("get_order", &get_order,
          py::arg("starts"),
          py::arg("widths"),
          "Get the order of genomic ranges");
}
