// pyranges_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "superintervals.hpp"

namespace py = pybind11;

template<typename S, typename T>
class PySuperIntervals : public SuperIntervals<S, T> {
public:
    PySuperIntervals() : SuperIntervals<S, T>() {}

    void add_from_arrays(py::array_t<S> starts, py::array_t<S> ends, py::array_t<T> data) {
        auto starts_buf = starts.request();
        auto ends_buf = ends.request();
        auto data_buf = data.request();

        if (starts_buf.size != ends_buf.size || starts_buf.size != data_buf.size) {
            throw std::runtime_error("Input arrays must have same length");
        }

        S* starts_ptr = static_cast<S*>(starts_buf.ptr);
        S* ends_ptr = static_cast<S*>(ends_buf.ptr);
        T* data_ptr = static_cast<T*>(data_buf.ptr);

        for (size_t i = 0; i < starts_buf.size; ++i) {
            this->add(starts_ptr[i], ends_ptr[i] - starts_ptr[i], data_ptr[i]);
        }
    }

    py::tuple find_overlaps(py::array_t<S> query_starts, py::array_t<S> query_ends) {
        auto starts_buf = query_starts.request();
        auto ends_buf = query_ends.request();

        if (starts_buf.size != ends_buf.size) {
            throw std::runtime_error("Query start and end arrays must have same length");
        }

        S* starts_ptr = static_cast<S*>(starts_buf.ptr);
        S* ends_ptr = static_cast<S*>(ends_buf.ptr);

        std::vector<std::vector<size_t>> all_overlaps;
        std::vector<std::vector<T>> all_data;

        for (size_t i = 0; i < starts_buf.size; ++i) {
            std::vector<T> overlaps;
            this->findOverlaps(starts_ptr[i], ends_ptr[i], overlaps);
            all_data.push_back(overlaps);

            // Get corresponding indices
            std::vector<size_t> indices;
            for (const auto& val : overlaps) {
                for (size_t j = 0; j < this->data.size(); ++j) {
                    if (this->data[j] == val) {
                        indices.push_back(j);
                        break;
                    }
                }
            }
            all_overlaps.push_back(indices);
        }

        return py::make_tuple(all_overlaps, all_data);
    }

    size_t count_overlaps(S start, S end) {
        return this->countOverlaps(start, end);
    }

    bool any_overlaps(S start, S end) {
        return this->anyOverlaps(start, end);
    }

    py::array_t<S> get_starts() {
        return py::array_t<S>(this->starts.size(), this->starts.data());
    }

    py::array_t<S> get_ends() {
        std::vector<S> ends;
        ends.reserve(this->starts.size());
        for (size_t i = 0; i < this->starts.size(); ++i) {
            ends.push_back(this->starts[i] + (this->ends[i] - this->starts[i]));  // Calculate end from start and width
        }
        return py::array_t<S>(ends.size(), ends.data());
    }

    py::array_t<S> get_widths() {
        std::vector<S> widths;
        widths.reserve(this->starts.size());
        for (size_t i = 0; i < this->starts.size(); ++i) {
            widths.push_back(this->ends[i] - this->starts[i]);  // Calculate width from start and end
        }
        return py::array_t<S>(widths.size(), widths.data());
    }

    py::array_t<T> get_data() {
        return py::array_t<T>(this->data.size(), this->data.data());
    }
};

PYBIND11_MODULE(lib_iranges, m) {
    py::class_<PySuperIntervals<int32_t, int32_t>>(m, "SuperIntervals")
        .def(py::init<>())
        .def("add_from_arrays", &PySuperIntervals<int32_t, int32_t>::add_from_arrays)
        .def("find_overlaps", &PySuperIntervals<int32_t, int32_t>::find_overlaps)
        .def("count_overlaps", &PySuperIntervals<int32_t, int32_t>::count_overlaps)
        .def("any_overlaps", &PySuperIntervals<int32_t, int32_t>::any_overlaps)
        .def("index", &PySuperIntervals<int32_t, int32_t>::index)
        .def("get_starts", &PySuperIntervals<int32_t, int32_t>::get_starts)
        .def("get_ends", &PySuperIntervals<int32_t, int32_t>::get_ends)
        .def("get_widths", &PySuperIntervals<int32_t, int32_t>::get_widths)
        .def("get_data", &PySuperIntervals<int32_t, int32_t>::get_data);
}
