#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

static py::array_t<double> coverage_sort(
    const py::array_t<int32>& starts,
    const py::array_t<int32>& widths,
    const py::array_t<double>& weight,
    int32 cvg_len
);

static py::array_t<double> coverage_hash(
    const py::array_t<int32>& starts,
    const py::array_t<int32>& widths,
    const py::array_t<double>& weight,
    int32 cvg_len
);

static py::array_t<double> coverage_naive(
    const py::array_t<int32>& starts,
    const py::array_t<int32>& widths,
    const py::array_t<double>& weight,
    int32 cvg_len
);

// exported to Python
// based on https://github.com/Bioconductor/IRanges/blob/devel/src/coverage_methods.c
std::tuple<py::array_t<int32>, py::array_t<int32>, int32, bool>
shift_and_clip_ranges(
    py::array_t<int32> starts,
    py::array_t<int32> widths,
    py::array_t<int32> shift,
    py::object width_obj,
    py::object circle_len_obj
) {
    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    auto shift_r = shift.unchecked<1>();

    int x_len = starts_r.shape(0);
    int shift_len = shift_r.shape(0);

    // Handle empty input
    if (x_len == 0) {
        int width_val = width_obj.is_none() ? 0 : width_obj.cast<int32>();
        return std::make_tuple(
            py::array_t<int32>(),
            py::array_t<int32>(),
            width_val,
            true
        );
    }

    if (shift_len == 0) {
        throw std::runtime_error("shift length must be > 0");
    }

    bool width_is_none = width_obj.is_none();
    bool circle_len_is_none = circle_len_obj.is_none();
    int32 width = width_is_none ? -1 : width_obj.cast<int32>();
    int32 circle_len = circle_len_is_none ? -1 : circle_len_obj.cast<int32>();

    bool auto_cvg_len = width_is_none || (!circle_len_is_none && circle_len > 0);
    int32 cvg_len = auto_cvg_len ? 0 : width;

    auto shifted_starts = py::array_t<int32>(x_len);
    auto new_widths = py::array_t<int32>(x_len);
    auto shifted_starts_ptr = shifted_starts.mutable_unchecked<1>();
    auto new_widths_ptr = new_widths.mutable_unchecked<1>();

    // Process ranges
    std::vector<std::pair<int32, int32>> sorted_ranges;
    sorted_ranges.reserve(x_len);

    for (int i = 0; i < x_len; i++) {
        int j = i % shift_len;
        int32 x_start = starts_r(i);
        int32 x_end = x_start + widths_r(i) - 1;
        int32 shift_val = shift_r(j);

        x_start += shift_val;
        x_end += shift_val;

        if (!circle_len_is_none) {
            if (circle_len <= 0) {
                throw std::runtime_error("circle_len must be > 0");
            }
            if (!width_is_none && width > circle_len) {
                throw std::runtime_error("width cannot be greater than circle_len");
            }

            int tmp = x_start % circle_len;
            if (tmp <= 0) tmp += circle_len;
            x_end += tmp - x_start;
            x_start = tmp;
        }

        // Clip ranges
        if (x_end < 0) {
            x_end = 0;
        } else if (x_end > cvg_len) {
            if (auto_cvg_len) {
                cvg_len = x_end;
            } else {
                x_end = cvg_len;
            }
        }

        if (x_start < 1) x_start = 1;
        else if (x_start > cvg_len + 1) x_start = cvg_len + 1;

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
        } else {
            for (size_t i = 1; i < sorted_ranges.size(); i++) {
                if (sorted_ranges[i].first != sorted_ranges[i-1].second + 1) {
                    out_ranges_are_tiles = false;
                    break;
                }
            }
        }
    }

    return std::make_tuple(shifted_starts, new_widths, cvg_len, out_ranges_are_tiles);
}

py::array_t<double> coverage(
    py::array_t<int32> starts,
    py::array_t<int32> widths,
    py::array_t<int32> shift,
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
        } else if (weight.shape(0) == x_len) {
            auto result = py::array_t<double>(cvg_len);
            auto result_ptr = result.mutable_unchecked<1>();
            auto weight_r = weight.unchecked<1>();
            auto widths_r = new_widths.unchecked<1>();

            int pos = 0;
            for (int i = 0; i < x_len; i++) {
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
    } else if (method == "hash") {
        return coverage_hash(shifted_starts, new_widths, weight, cvg_len);
    } else {
        return coverage_naive(shifted_starts, new_widths, weight, cvg_len);
    }
}

static py::array_t<double> coverage_sort(
    const py::array_t<int32>& starts,
    const py::array_t<int32>& widths,
    const py::array_t<double>& weight,
    int32 cvg_len
) {
    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    auto weight_r = weight.unchecked<1>();
    int x_len = starts_r.shape(0);

    std::vector<std::tuple<int32, int32, double>> events;
    events.reserve(2 * x_len);

    for (int i = 0; i < x_len; i++) {
        int32 start = starts_r(i);
        int32 width = widths_r(i);
        double w = weight_r(i % weight_r.shape(0));
        events.emplace_back(start, 1, w);
        events.emplace_back(start + width, -1, w);
    }

    std::sort(events.begin(), events.end());

    auto coverage = py::array_t<double>(cvg_len);
    auto coverage_ptr = coverage.mutable_unchecked<1>();
    double current_sum = 0;
    int prev_pos = 1;

    for (const auto& event : events) {
        int pos = std::get<0>(event);
        if (pos > cvg_len) break;

        if (pos > prev_pos) {
            for (int i = prev_pos - 1; i < pos - 1; i++) {
                coverage_ptr(i) = current_sum;
            }
        }

        current_sum += std::get<1>(event) * std::get<2>(event);
        prev_pos = pos;
    }

    if (prev_pos <= cvg_len) {
        for (int i = prev_pos - 1; i < cvg_len; i++) {
            coverage_ptr(i) = current_sum;
        }
    }

    return coverage;
}

static py::array_t<double> coverage_hash(
    const py::array_t<int32>& starts,
    const py::array_t<int32>& widths,
    const py::array_t<double>& weight,
    int32 cvg_len
) {
    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    auto weight_r = weight.unchecked<1>();
    int x_len = starts_r.shape(0);

    std::vector<double> cvg_buf(cvg_len + 1, 0.0);

    for (int i = 0; i < x_len; i++) {
        int32 start = starts_r(i);
        int32 width = widths_r(i);
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

    for (int i = 0; i < cvg_len; i++) {
        cumsum += cvg_buf[i];
        result_ptr(i) = cumsum;
    }

    return result;
}

static py::array_t<double> coverage_naive(
    const py::array_t<int32>& starts,
    const py::array_t<int32>& widths,
    const py::array_t<double>& weight,
    int32 cvg_len
) {
    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    auto weight_r = weight.unchecked<1>();
    int x_len = starts_r.shape(0);

    auto coverage = py::array_t<double>(cvg_len);
    auto coverage_ptr = coverage.mutable_unchecked<1>();
    std::fill_n(coverage_ptr.mutable_data(0), cvg_len, 0.0);

    for (int i = 0; i < x_len; i++) {
        int32 start = starts_r(i);
        int32 width = widths_r(i);
        double w = weight_r(i % weight_r.shape(0));

        for (int j = start - 1; j < start + width - 1 && j < cvg_len; j++) {
            if (j >= 0) {
                coverage_ptr(j) += w;
            }
        }
    }

    return coverage;
}

void init_coverage(pybind11::module& m) {
    m.def("shift_and_clip_ranges", &shift_and_clip_ranges,
          py::arg("starts"),
          py::arg("widths"),
          py::arg("shift"),
          py::arg("width"),
          py::arg("circle_len"),
          "Shift and clip genomic ranges");

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
