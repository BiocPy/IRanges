#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <tuple>
#include <memory>
#include <stdexcept>
#include "nclist/nclist.hpp"
#include <thread>
#include <numeric>
#include <cstdint>

namespace py = pybind11;

// Using integer types for consistency with numpy
using Index = std::int32_t;
using Position = std::int32_t;
struct NCListHandler {
    NCListHandler(py::array_t<Position> starts, py::array_t<Position> ends) {
        py::buffer_info starts_buf = starts.request();
        py::buffer_info ends_buf = ends.request();

        auto starts_ptr = static_cast<const Position*>(starts_buf.ptr);
        auto ends_ptr = static_cast<const Position*>(ends_buf.ptr);
        Index n = starts_buf.shape[0];

        nclist_obj = nclist::build<Index, Position>(n, starts_ptr, ends_ptr);
    }

    nclist::Nclist<Index, Position> nclist_obj;
};

pybind11::tuple perform_find_overlaps(
    NCListHandler &self,
    py::array_t<Position> query_starts,
    py::array_t<Position> query_ends,
    const std::string& query_type,
    const std::string& select,
    int max_gap,
    int min_overlap,
    int num_threads=1) {

    py::buffer_info q_starts_buf = query_starts.request();
    py::buffer_info q_ends_buf = query_ends.request();

    auto q_starts_ptr = static_cast<const Position*>(q_starts_buf.ptr);
    auto q_ends_ptr = static_cast<const Position*>(q_ends_buf.ptr);
    Index n_queries = q_starts_buf.shape[0];

    bool quit_on_first = (select == "first" || select == "arbitrary");
    if (select != "all" && select != "last" && !quit_on_first) {
        throw std::runtime_error("Invalid 'select' parameter. Must be 'all', 'first', 'last', or 'arbitrary'.");
    }

    std::vector<std::vector<Index> > all_results(n_queries);
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    int num_jobs = n_queries / num_threads;
    int num_remaining = n_queries % num_threads;
    int jobs_so_far = 0;
    for (int i = 0; i < num_threads; ++i) {
        int current_jobs = num_jobs + (i < num_remaining);

        if (current_jobs == 0) {
            break;
        }

        workers.emplace_back([&](int first, int length) -> void {
            std::vector<Index> single_query_matches;

            for (Index i = first, last = first + length; i < last; ++i) {
                if (query_type == "any") {
                    nclist::OverlapsAnyWorkspace<Index> ws_any;
                    nclist::OverlapsAnyParameters<Position> params;
                    params.min_overlap = min_overlap > 0 ? min_overlap : 0;
                    if (max_gap >= 0) params.max_gap = max_gap;
                    params.quit_on_first = quit_on_first;
                    nclist::overlaps_any(self.nclist_obj, q_starts_ptr[i], q_ends_ptr[i], params, ws_any, single_query_matches);
                } else if (query_type == "start") {
                    nclist::OverlapsStartWorkspace<Index> ws_start;
                    nclist::OverlapsStartParameters<Position> params;
                    params.min_overlap = min_overlap > 0 ? min_overlap : 0;
                    if (max_gap >= 0) params.max_gap = max_gap;
                    params.quit_on_first = quit_on_first;
                    nclist::overlaps_start(self.nclist_obj, q_starts_ptr[i], q_ends_ptr[i], params, ws_start, single_query_matches);
                } else if (query_type == "end") {
                    nclist::OverlapsEndWorkspace<Index> ws_end;
                    nclist::OverlapsEndParameters<Position> params;
                    params.min_overlap = min_overlap > 0 ? min_overlap : 0;
                    if (max_gap >= 0) params.max_gap = max_gap;
                    params.quit_on_first = quit_on_first;
                    nclist::overlaps_end(self.nclist_obj, q_starts_ptr[i], q_ends_ptr[i], params, ws_end, single_query_matches);
                } else if (query_type == "within") {
                    nclist::OverlapsWithinWorkspace<Index> ws_within;
                    nclist::OverlapsWithinParameters<Position> params;
                    params.min_overlap = min_overlap > 0 ? min_overlap : 0;
                    if (max_gap >= 0) {
                        params.max_gap = max_gap;
                    }
                    params.quit_on_first = quit_on_first;
                    nclist::overlaps_within(self.nclist_obj, q_starts_ptr[i], q_ends_ptr[i], params, ws_within, single_query_matches);
                }

                if (!single_query_matches.empty()) {
                    if (select == "last" && !quit_on_first) {
                        all_results[i] = {single_query_matches.back()};
                    } else {
                        all_results[i] = single_query_matches;
                    }
                }
            }
        }, jobs_so_far, current_jobs);
        jobs_so_far += current_jobs;
    }

    for (auto& worker : workers) {
        worker.join();
    }

    std::size_t total_hits = 0;
    for(const auto& res : all_results) {
        total_hits += res.size();
    }

    py::array_t<Index> self_hits(total_hits);
    py::array_t<Index> query_hits(total_hits);
    auto s_res_ptr = static_cast<Index*>(self_hits.request().ptr);
    auto q_res_ptr = static_cast<Index*>(query_hits.request().ptr);

    std::size_t current_pos = 0;
    for (Index i = 0; i < n_queries; ++i) {
        if (!all_results[i].empty()) {
            std::copy(all_results[i].begin(), all_results[i].end(), s_res_ptr + current_pos);
            std::fill(q_res_ptr + current_pos, q_res_ptr + current_pos + all_results[i].size(), i);
            current_pos += all_results[i].size();
        }
    }

    return py::make_tuple(self_hits, query_hits);
}


struct GroupInfo {
    const Index* ptr;
    std::size_t size;
};

pybind11::tuple perform_find_overlaps_groups(
    py::array_t<Position> self_starts,
    py::array_t<Position> self_ends,
    const std::vector<py::array_t<Index>>& self_groups,
    py::array_t<Position> query_starts,
    py::array_t<Position> query_ends,
    const std::vector<py::array_t<Index>>& query_groups,
    const std::string& query_type,
    const std::string& select,
    int max_gap,
    int min_overlap,
    int num_threads=1) {

    auto s_starts_ptr = static_cast<const Position*>(self_starts.request().ptr);
    auto s_ends_ptr = static_cast<const Position*>(self_ends.request().ptr);
    auto q_starts_ptr = static_cast<const Position*>(query_starts.request().ptr);
    auto q_ends_ptr = static_cast<const Position*>(query_ends.request().ptr);
    std::size_t n_groups = self_groups.size();
    if (n_groups != query_groups.size()) {
        throw std::runtime_error("The number of self/subject groups must be equal to the number of query groups.");
    }

    bool quit_on_first = (select == "first" || select == "arbitrary");
    if (select != "all" && select != "last" && !quit_on_first) {
        throw std::runtime_error("Invalid 'select' parameter. Must be 'all', 'first', 'last', or 'arbitrary'.");
    }

    std::vector<GroupInfo> self_group_info(n_groups);
    std::vector<GroupInfo> query_group_info(n_groups);
    for (std::size_t i = 0; i < n_groups; ++i) {
        auto s_req = self_groups[i].request();
        self_group_info[i] = {static_cast<const Index*>(s_req.ptr), static_cast<std::size_t>(s_req.shape[0])};
        auto q_req = query_groups[i].request();
        query_group_info[i] = {static_cast<const Index*>(q_req.ptr), static_cast<std::size_t>(q_req.shape[0])};
    }

    std::vector<std::vector<std::vector<Index> > > all_group_results(n_groups);
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    int num_jobs = n_groups / num_threads;
    int num_remaining = n_groups % num_threads;
    int jobs_so_far = 0;

    for (int i = 0; i < num_threads; ++i) {
        int current_jobs = num_jobs + (i < num_remaining);
        if (current_jobs == 0) {
            break;
        }

        workers.emplace_back([&](int first, int length) -> void {
            std::vector<Index> single_query_matches;

            for (Index j = first, last = first + length; j < last; ++j) {

                const auto& s_info = self_group_info[j];
                const auto& q_info = query_group_info[j];

                if (s_info.size == 0 || q_info.size == 0) {
                    continue;
                }

                auto nclist_obj = nclist::build<Index, Position>(s_info.size, s_info.ptr, s_starts_ptr, s_ends_ptr);
                std::vector<std::vector<Index> > local_group_results;
                local_group_results.resize(q_info.size);

                for (std::size_t k = 0; k < q_info.size; ++k) {
                    Index original_query_idx = q_info.ptr[k];

                    if (query_type == "any") {
                        nclist::OverlapsAnyWorkspace<Index> ws_any;
                        nclist::OverlapsAnyParameters<Position> params;
                        params.min_overlap = min_overlap > 0 ? min_overlap : 0;
                        if (max_gap >= 0) params.max_gap = max_gap;
                        params.quit_on_first = quit_on_first;
                        nclist::overlaps_any(nclist_obj, q_starts_ptr[original_query_idx], q_ends_ptr[original_query_idx], params, ws_any, single_query_matches);
                    } else if (query_type == "start") {
                        nclist::OverlapsStartWorkspace<Index> ws_start;
                        nclist::OverlapsStartParameters<Position> params;
                        params.min_overlap = min_overlap > 0 ? min_overlap : 0;
                        if (max_gap >= 0) params.max_gap = max_gap;
                        params.quit_on_first = quit_on_first;
                        nclist::overlaps_start(nclist_obj, q_starts_ptr[original_query_idx], q_ends_ptr[original_query_idx], params, ws_start, single_query_matches);
                    } else if (query_type == "end") {
                        nclist::OverlapsEndWorkspace<Index> ws_end;
                        nclist::OverlapsEndParameters<Position> params;
                        params.min_overlap = min_overlap > 0 ? min_overlap : 0;
                        if (max_gap >= 0) params.max_gap = max_gap;
                        params.quit_on_first = quit_on_first;
                        nclist::overlaps_end(nclist_obj, q_starts_ptr[original_query_idx], q_ends_ptr[original_query_idx], params, ws_end, single_query_matches);
                    } else if (query_type == "within") {
                        nclist::OverlapsWithinWorkspace<Index> ws_within;
                        nclist::OverlapsWithinParameters<Position> params;
                        params.min_overlap = min_overlap > 0 ? min_overlap : 0;
                        if (max_gap >= 0) {
                            params.max_gap = max_gap;
                        }
                        params.quit_on_first = quit_on_first;
                        nclist::overlaps_within(nclist_obj, q_starts_ptr[original_query_idx], q_ends_ptr[original_query_idx], params, ws_within, single_query_matches);
                    }

                    if (!single_query_matches.empty()) {
                         if (select == "last" && !quit_on_first) {
                            local_group_results[k] = {single_query_matches.back()};
                        } else {
                            local_group_results[k] = single_query_matches;
                        }
                    }
                }

                if (!local_group_results.empty()){
                    all_group_results[j] = std::move(local_group_results);
                }
            }
        }, jobs_so_far, current_jobs);

        jobs_so_far += current_jobs;
    }

    for (auto& worker : workers) {
        worker.join();
    }

    std::size_t total_hits = 0;
    for(const auto& group_res : all_group_results) {
        for (const auto& query_res : group_res) {
            total_hits += query_res.size();
        }
    }

    py::array_t<Index> query_hits(total_hits);
    py::array_t<Index> self_hits(total_hits);
    auto q_res_ptr = static_cast<Index*>(query_hits.request().ptr);
    auto s_res_ptr = static_cast<Index*>(self_hits.request().ptr);

    std::size_t current_pos = 0;
    for (std::size_t group_idx = 0, all_group_size=all_group_results.size(); group_idx < all_group_size; ++group_idx) {
        const auto& group_res = all_group_results[group_idx];
        const auto& q_info = query_group_info[group_idx];

        for (std::size_t query_in_group_idx = 0, group_res_size = group_res.size(); query_in_group_idx < group_res_size; ++query_in_group_idx) {
            const auto& relative_subject_matches = group_res[query_in_group_idx];
            if (!relative_subject_matches.empty()) {
                Index original_query_idx = q_info.ptr[query_in_group_idx];
                std::copy(relative_subject_matches.begin(), relative_subject_matches.end(), s_res_ptr + current_pos);
                std::fill(q_res_ptr + current_pos, q_res_ptr + current_pos + relative_subject_matches.size(), original_query_idx);
                current_pos += relative_subject_matches.size();
            }
        }
    }

    return py::make_tuple(self_hits, query_hits);
}


void init_nclist(pybind11::module &m){

    py::class_<NCListHandler>(m, "NCListHandler", "Manages an nclist-cpp index for overlap queries.")
        .def(py::init<py::array_t<Position>, py::array_t<Position>>(),
            py::arg("starts"), py::arg("ends"))

        .def("find_overlaps", &perform_find_overlaps,
            py::arg("query_starts"),
            py::arg("query_ends"),
            py::arg("query_type") = "any",
            py::arg("select") = "all",
            py::arg("max_gap") = -1,
            py::arg("min_overlap") = 1,
            py::arg("num_threads") = 1,
            "Finds overlaps between query intervals and the indexed subject intervals.");

    m.def("find_overlaps_groups", &perform_find_overlaps_groups,
        py::arg("self_starts"),
        py::arg("self_ends"),
        py::arg("self_groups"),
        py::arg("query_starts"),
        py::arg("query_ends"),
        py::arg("query_groups"),
        py::arg("query_type") = "any",
        py::arg("select") = "all",
        py::arg("max_gap") = -1,
        py::arg("min_overlap") = 1,
        py::arg("num_threads") = 1,
        "Finds overlaps between query and subject intervals, respecting group boundaries.");
}
