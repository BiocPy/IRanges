#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <tuple>

namespace py = pybind11;

// Similar implementations to
// https://github.com/Bioconductor/IRanges/blob/devel/src/inter_range_methods.c

static std::vector<int32_t> get_order(
    const py::array_t<int32_t> &starts,
    const py::array_t<int32_t> &widths
) {

    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    int n = starts_r.shape(0);

    std::vector<int> order(n);
    for (int i = 0; i < n; i++) {
        order[i] = i;
    }

    // Sort by start position, then by width
    std::sort(order.begin(), order.end(), [&starts_r, &widths_r](int32_t i, int32_t j) {
        if (starts_r(i) != starts_r(j)) {
            return starts_r(i) < starts_r(j);
        }
        return widths_r(i) < widths_r(j);
    });

    return order;
}

py::dict reduce_ranges(
    py::array_t<int32_t> starts,
    py::array_t<int32_t> widths,
    bool drop_empty_ranges,
    int32_t min_gapwidth,
    bool with_revmap,
    bool with_inframe_start
) {

    if (min_gapwidth < 0) {
        throw std::runtime_error("negative min_gapwidth not supported");
    }

    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    int x_len = starts_r.shape(0);

    std::vector<int> order = get_order(starts, widths);

    std::vector<int32_t> out_starts;
    std::vector<int32_t> out_widths;
    std::vector<std::vector<int32_t>> revmap;
    std::vector<int32_t> inframe_start(with_inframe_start ? x_len : 0);

    int out_len = 0;
    int32_t max_end = 0;
    int32_t delta = 0;
    bool append_or_drop = true;

    for (int i = 0; i < x_len; i++) {
        int j = order[i];
        int32_t start_j = starts_r(j);
        int32_t width_j = widths_r(j);
        int32_t end_j = start_j + width_j - 1;

        if (i == 0) {
            append_or_drop = true;
            max_end = end_j;
            delta = start_j - 1;
        }
        else {
            int32_t gapwidth = start_j - max_end - 1;
            if (gapwidth >= min_gapwidth) {
                append_or_drop = true;
            }
        }

        if (append_or_drop) {
            if (width_j != 0 || (!drop_empty_ranges && (out_len == 0 || start_j != out_starts.back()))) {
                // Append new range
                out_starts.push_back(start_j);
                out_widths.push_back(width_j);
                if (with_revmap) {
                    revmap.emplace_back(1, j);
                }
                out_len++;
                append_or_drop = false;
            }
            max_end = end_j;
            if (i != 0) {
                delta += start_j - max_end - 1;
            }
        }
        else {
            int32_t width_inc = end_j - max_end;
            if (width_inc > 0) {
                // Merge with last range
                out_widths.back() += width_inc;
                max_end = end_j;
            }
            if (!(width_j == 0 && drop_empty_ranges) && with_revmap) {
                revmap.back().push_back(j);
            }
        }

        if (with_inframe_start) {
            inframe_start[j] = start_j - delta;
        }
    }

    py::dict result;
    auto out_starts_arr = py::array_t<int32_t>(out_starts.size());
    auto out_widths_arr = py::array_t<int32_t>(out_widths.size());

    auto out_starts_ptr = out_starts_arr.mutable_unchecked<1>();
    auto out_widths_ptr = out_widths_arr.mutable_unchecked<1>();

    for (size_t i = 0; i < out_starts.size(); i++) {
        out_starts_ptr(i) = out_starts[i];
        out_widths_ptr(i) = out_widths[i];
    }

    result["start"] = out_starts_arr;
    result["width"] = out_widths_arr;

    if (with_revmap) {
        // Convert revmap to list of numpy arrays
        py::list revmap_list;
        for (const auto &mapping : revmap) {
            auto map_arr = py::array_t<int32_t>(mapping.size());
            auto map_ptr = map_arr.mutable_unchecked<1>();
            for (size_t i = 0; i < mapping.size(); i++) {
                map_ptr(i) = mapping[i];
            }
            revmap_list.append(map_arr);
        }
        result["revmap"] = revmap_list;
    }
    else {
        result["revmap"] = py::none();
    }

    if (with_inframe_start) {
        auto inframe_arr = py::array_t<int32_t>(inframe_start.size());
        auto inframe_ptr = inframe_arr.mutable_unchecked<1>();
        for (size_t i = 0; i < inframe_start.size(); i++) {
            inframe_ptr(i) = inframe_start[i];
        }
        result["inframe.start"] = inframe_arr;
    }
    else {
        result["inframe.start"] = py::none();
    }

    return result;
}

std::tuple<py::array_t<int32_t>, py::array_t<int32_t>> gaps_ranges(
    py::array_t<int32_t> starts,
    py::array_t<int32_t> widths,
    py::object restrict_start_obj,
    py::object restrict_end_obj
) {

    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    int x_len = starts_r.shape(0);

    bool has_restrict_start = !restrict_start_obj.is_none();
    bool has_restrict_end = !restrict_end_obj.is_none();
    int32_t restrict_start = has_restrict_start ? restrict_start_obj.cast<int32_t>() : std::numeric_limits<int32_t>::min();
    int32_t restrict_end = has_restrict_end ? restrict_end_obj.cast<int32_t>() : std::numeric_limits<int32_t>::max();

    std::vector<int> order = get_order(starts, widths);

    std::vector<int32_t> gap_starts;
    std::vector<int32_t> gap_widths;

    int32_t max_end = has_restrict_start ? restrict_start - 1 : std::numeric_limits<int32_t>::min();

    for (int i = 0; i < x_len; i++) {
        int j = order[i];
        int32_t width_j = widths_r(j);

        if (width_j == 0)
            continue;

        int32_t start_j = starts_r(j);
        int32_t end_j = start_j + width_j - 1;

        if (max_end == std::numeric_limits<int32_t>::min()) {
            max_end = end_j;
        }
        else {
            int32_t gap_start = max_end + 1;

            if (has_restrict_end && start_j > restrict_end + 1) {
                start_j = restrict_end + 1;
            }

            int32_t gap_width = start_j - gap_start;

            if (gap_width >= 1) {
                // Add gap to output
                gap_starts.push_back(gap_start);
                gap_widths.push_back(gap_width);
                max_end = end_j;
            }
            else if (end_j > max_end) {
                max_end = end_j;
            }
        }

        if (has_restrict_end && max_end >= restrict_end)
            break;
    }

    // Handle final gap
    if (has_restrict_end &&
        max_end != std::numeric_limits<int32_t>::min() &&
        max_end < restrict_end) {
        int32_t gap_start = max_end + 1;
        int32_t gap_width = restrict_end - max_end;
        gap_starts.push_back(gap_start);
        gap_widths.push_back(gap_width);
    }

    auto out_starts = py::array_t<int32_t>(gap_starts.size());
    auto out_widths = py::array_t<int32_t>(gap_widths.size());

    auto out_starts_ptr = out_starts.mutable_unchecked<1>();
    auto out_widths_ptr = out_widths.mutable_unchecked<1>();

    for (size_t i = 0; i < gap_starts.size(); i++) {
        out_starts_ptr(i) = gap_starts[i];
        out_widths_ptr(i) = gap_widths[i];
    }

    return std::make_tuple(out_starts, out_widths);
}

py::array_t<int32_t> disjoint_bins(
    py::array_t<int32_t> starts,
    py::array_t<int32_t> widths
) {

    auto starts_r = starts.unchecked<1>();
    auto widths_r = widths.unchecked<1>();
    int n = starts_r.shape(0);

    auto result = py::array_t<int32_t>(n);
    auto result_ptr = result.mutable_unchecked<1>();

    std::vector<int32_t> bin_ends;
    bin_ends.reserve(128);

    for (int i = 0; i < n; i++) {
        int32_t start = starts_r(i);
        int32_t end = start + widths_r(i) - 1;

        // Find appropriate bin
        int j = 0;
        for (; j < static_cast<int>(bin_ends.size()) && bin_ends[j] >= start; j++);

        if (j == static_cast<int>(bin_ends.size())) {
            bin_ends.push_back(end);
        } else {
            bin_ends[j] = end;
        }

        result_ptr(i) = j;
    }

    return result;
}


void init_interranges(pybind11::module &m) {
    m.def("get_order", &get_order,
          py::arg("starts"),
          py::arg("widths"),
          "Get the order of genomic ranges");

    m.def("reduce_ranges", &reduce_ranges,
          py::arg("starts"),
          py::arg("widths"),
          py::arg("drop_empty_ranges") = false,
          py::arg("min_gapwidth") = 0,
          py::arg("with_revmap") = false,
          py::arg("with_inframe_start") = false,
          "Reduce ranges by merging overlapping or adjacent ranges");

    m.def("gaps_ranges", &gaps_ranges,
          py::arg("starts"),
          py::arg("widths"),
          py::arg("restrict_start") = py::none(),
          py::arg("restrict_end") = py::none(),
          "Find gaps between ranges");

    m.def("disjoint_bins", &disjoint_bins,
          py::arg("starts"),
          py::arg("widths"),
          "Assign ranges to disjoint bins");
}
