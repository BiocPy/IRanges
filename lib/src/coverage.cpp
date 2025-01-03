#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

// Similar implementations to
// based on https://github.com/Bioconductor/IRanges/blob/devel/src/coverage_methods.c

std::tuple<py::array_t<int32_t>, py::array_t<int32_t>, int32_t, bool>
shift_and_clip_ranges(
    py::array_t<int32_t> starts,
    py::array_t<int32_t> widths,
    py::array_t<int32_t> shift,
    py::object width_obj,
    py::object circle_len_obj
) {

    // // validate?
    // if (starts.dtype() != py::dtype::of<int32_t>() ||
    //     widths.dtype() != py::dtype::of<int32_t>() ||
    //     shift.dtype() != py::dtype::of<int32_t>()) {
    //     throw std::runtime_error("All inputs must be int32");
    // }

    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    auto shift_r = shift.unchecked<1>();

    int x_len = starts_r.shape(0);
    int shift_len = shift_r.shape(0);

    if (x_len == 0) {
        int32_t width_val = width_obj.is_none() ? 0 : width_obj.cast<int32_t>();
        return std::make_tuple(
            py::array_t<int32_t>(),
            py::array_t<int32_t>(),
            width_val,
            true);
    }

    if (shift_len == 0) {
        throw std::runtime_error("shift length must be > 0");
    }

    bool width_is_none = width_obj.is_none();
    bool circle_len_is_none = circle_len_obj.is_none();
    int32_t width = width_is_none ? -1 : width_obj.cast<int32_t>();
    int32_t circle_len = circle_len_is_none ? -1 : circle_len_obj.cast<int32_t>();

    bool auto_cvg_len = width_is_none || (!circle_len_is_none && circle_len > 0);
    int32_t cvg_len = auto_cvg_len ? 0 : width;

    auto shifted_starts = py::array_t<int32_t>(x_len);
    auto new_widths = py::array_t<int32_t>(x_len);
    auto shifted_starts_ptr = shifted_starts.mutable_unchecked<1>();
    auto new_widths_ptr = new_widths.mutable_unchecked<1>();

    std::vector<std::pair<int32_t, int32_t>> sorted_ranges;
    sorted_ranges.reserve(x_len);

    for (int i = 0; i < x_len; i++) {
        int j = i % shift_len;
        int32_t x_start = starts_r(i);
        int32_t x_end = x_start + widths_r(i) - 1;
        int32_t shift_val = shift_r(j);

        x_start += shift_val;
        x_end += shift_val;

        // Handle circular sequence
        if (!circle_len_is_none) {
            if (circle_len <= 0) {
                throw std::runtime_error("circle_len must be > 0");
            }
            if (!width_is_none && width > circle_len) {
                throw std::runtime_error("width cannot be greater than circle_len");
            }

            int32_t tmp = x_start % circle_len;
            if (tmp <= 0)
                tmp += circle_len;
            x_end += tmp - x_start;
            x_start = tmp;
        }

        if (x_end < 0) {
            x_end = 0;
        }
        else if (x_end > cvg_len) {
            if (auto_cvg_len) {
                cvg_len = x_end;
            }
            else {
                x_end = cvg_len;
            }
        }

        if (x_start < 1)
            x_start = 1;
        else if (x_start > cvg_len + 1)
            x_start = cvg_len + 1;

        shifted_starts_ptr(i) = x_start;
        new_widths_ptr(i) = x_end - x_start + 1;
        sorted_ranges.emplace_back(x_start, x_end);
    }

    // Check tiling configuration
    bool out_ranges_are_tiles = true;
    if (x_len > 0) {
        std::sort(sorted_ranges.begin(), sorted_ranges.end());
        if (sorted_ranges[0].first != 1 || sorted_ranges.back().second != cvg_len) {
            out_ranges_are_tiles = false;
        }
        else {
            for (size_t i = 1; i < sorted_ranges.size(); i++) {
                if (sorted_ranges[i].first != sorted_ranges[i - 1].second + 1) {
                    out_ranges_are_tiles = false;
                    break;
                }
            }
        }
    }

    return std::make_tuple(shifted_starts, new_widths, cvg_len, out_ranges_are_tiles);
}

static py::array_t<double> coverage_sort(
    const py::array_t<int32_t> &starts,
    const py::array_t<int32_t> &widths,
    const py::array_t<double> &weight,
    int32_t cvg_len
) {
    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    auto weight_r = weight.unchecked<1>();
    int x_len = starts_r.shape(0);

    std::vector<std::tuple<int32_t, int, double>> events;
    events.reserve(2 * x_len);

    for (int i = 0; i < x_len; i++) {
        int32_t start = starts_r(i);
        int32_t width = widths_r(i);
        double w = weight_r(i % weight_r.shape(0));
        events.emplace_back(start, 1, w);
        events.emplace_back(start + width, -1, w);
    }

    std::sort(events.begin(), events.end());

    auto coverage = py::array_t<double>(cvg_len);
    auto coverage_ptr = coverage.mutable_unchecked<1>();
    double current_sum = 0;
    int32_t prev_pos = 1;

    for (const auto &event : events) {
        int32_t pos = std::get<0>(event);
        if (pos > cvg_len)
            break;

        if (pos > prev_pos) {
            for (int32_t i = prev_pos - 1; i < pos - 1; i++)
            {
                coverage_ptr(i) = current_sum;
            }
        }

        current_sum += std::get<1>(event) * std::get<2>(event);
        prev_pos = pos;
    }

    if (prev_pos <= cvg_len) {
        for (int32_t i = prev_pos - 1; i < cvg_len; i++) {
            coverage_ptr(i) = current_sum;
        }
    }

    return coverage;
}

static py::array_t<double> coverage_hash(
    const py::array_t<int32_t> &starts,
    const py::array_t<int32_t> &widths,
    const py::array_t<double> &weight,
    int32_t cvg_len
) {
    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    auto weight_r = weight.unchecked<1>();
    int x_len = starts_r.shape(0);

    std::vector<double> cvg_buf(cvg_len + 1, 0.0);

    for (int i = 0; i < x_len; i++) {
        int32_t start = starts_r(i);
        int32_t width = widths_r(i);
        double w = weight_r(i % weight_r.shape(0));

        if (width > 0) {
            cvg_buf[start - 1] += w;
            if (start + width - 1 <= cvg_len) {
                cvg_buf[start + width - 1] -= w;
            }
        }
    }

    auto result = py::array_t<double>(cvg_len);
    auto result_ptr = result.mutable_unchecked<1>();
    double cumsum = 0.0;

    for (int32_t i = 0; i < cvg_len; i++) {
        cumsum += cvg_buf[i];
        result_ptr(i) = cumsum;
    }

    return result;
}

static py::array_t<double> coverage_naive(
    const py::array_t<int32_t> &starts,
    const py::array_t<int32_t> &widths,
    const py::array_t<double> &weight,
    int32_t cvg_len
) {
    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    auto weight_r = weight.unchecked<1>();
    int x_len = starts_r.shape(0);

    auto coverage = py::array_t<double>(cvg_len);
    auto coverage_ptr = coverage.mutable_unchecked<1>();
    std::fill_n(coverage_ptr.mutable_data(0), cvg_len, 0.0);

    for (int i = 0; i < x_len; i++) {
        int32_t start = starts_r(i);
        int32_t width = widths_r(i);
        double w = weight_r(i % weight_r.shape(0));

        for (int32_t j = start - 1; j < start + width - 1 && j < cvg_len; j++) {
            if (j >= 0) {
                coverage_ptr(j) += w;
            }
        }
    }

    return coverage;
}

py::array_t<double> coverage(
    py::array_t<int32_t> starts,
    py::array_t<int32_t> widths,
    py::array_t<int32_t> shift,
    py::object width,
    py::array_t<double> weight,
    py::object circle_len,
    std::string method = "auto"
) {

    auto [shifted_starts, new_widths, cvg_len, out_ranges_are_tiles] =
        shift_and_clip_ranges(starts, widths, shift, width, circle_len);

    int x_len = shifted_starts.shape(0);

    if (x_len == 0 || cvg_len == 0) {
        auto result = py::array_t<double>(cvg_len);
        auto result_ptr = result.mutable_unchecked<1>();
        std::fill_n(result_ptr.mutable_data(0), cvg_len, 0.0);
        return result;
    }

    // Handle tiling case optimization
    if (out_ranges_are_tiles) {
        if (weight.shape(0) == 1) {
            auto result = py::array_t<double>(cvg_len);
            auto result_ptr = result.mutable_unchecked<1>();
            double w = weight.unchecked<1>()(0);
            std::fill_n(result_ptr.mutable_data(0), cvg_len, w);
            return result;
        }
        else if (weight.shape(0) == x_len) {
            auto result = py::array_t<double>(cvg_len);
            auto result_ptr = result.mutable_unchecked<1>();
            auto weight_r = weight.unchecked<1>();
            auto widths_r = new_widths.unchecked<1>();

            int pos = 0;
            for (int i = 0; i < x_len; i++)
            {
                std::fill_n(result_ptr.mutable_data(pos), widths_r(i), weight_r(i));
                pos += widths_r(i);
            }
            return result;
        }
    }

    if (method == "auto") {
        method = (x_len <= 0.25 * cvg_len) ? "sort" : "hash";
    }

    if (method == "sort") {
        return coverage_sort(shifted_starts, new_widths, weight, cvg_len);
    }
    else if (method == "hash") {
        return coverage_hash(shifted_starts, new_widths, weight, cvg_len);
    }
    else {
        return coverage_naive(shifted_starts, new_widths, weight, cvg_len);
    }
}

void init_coverage(pybind11::module &m) {
    m.def("shift_and_clip_ranges", &shift_and_clip_ranges,
          py::arg("starts"),
          py::arg("widths"),
          py::arg("shift"),
          py::arg("width"),
          py::arg("circle_len"),
          "Shift and clip ranges");

    m.def("coverage", &coverage,
          py::arg("starts"),
          py::arg("widths"),
          py::arg("shift"),
          py::arg("width"),
          py::arg("weight"),
          py::arg("circle_len"),
          py::arg("method") = "auto",
          "Compute weighted coverage of ranges");
}
