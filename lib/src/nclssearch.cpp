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

// Using specific integer types for consistency with numpy
using Index = std::int32_t;
using Position = std::int32_t;

struct NCListSearchHandler {
    NCListSearchHandler(py::array_t<Position> starts, py::array_t<Position> ends) {
        py::buffer_info starts_buf = starts.request();
        py::buffer_info ends_buf = ends.request();

        auto starts_ptr = static_cast<const Position*>(starts_buf.ptr);
        auto ends_ptr = static_cast<const Position*>(ends_buf.ptr);
        Index n = starts_buf.shape[0];

        nclist_obj = nclist::build<Index, Position>(n, starts_ptr, ends_ptr);
    }

    nclist::Nclist<Index, Position> nclist_obj;
};

pybind11::tuple perform_follow(
    NCListSearchHandler &self,
    py::array_t<Position> query_starts,
    py::array_t<Position> query_ends,
    const std::string& select,
    int num_threads = 1) {

    auto q_starts_ptr = static_cast<const Position*>(query_starts.request().ptr);
    auto q_ends_ptr = static_cast<const Position*>(query_ends.request().ptr);

    Index n_queries = query_starts.request().shape[0];

    bool quit_on_first = (select == "last");
    if (select != "all" && select != "last" && !quit_on_first) {
        throw std::runtime_error("Invalid 'select' parameter. Must be 'all' or 'last'.");
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
            std::vector<Index> nearest_matches;

            for (Index i = first, last = first + length; i < last; ++i) {
                nclist::NearestWorkspace<Index> ws_nearest;
                nclist::NearestParameters<Position> params;
                params.quit_on_first = quit_on_first;

                nclist::nearest(self.nclist_obj, q_starts_ptr[i], q_ends_ptr[i], params, ws_nearest, nearest_matches);

                if (!nearest_matches.empty()) {
                    if (select == "last" && !quit_on_first) {
                        all_results[i] = {nearest_matches.back()};
                    } else {
                        all_results[i] = nearest_matches;
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

pybind11::tuple perform_precede(
    NCListSearchHandler &self,
    py::array_t<Position> query_starts,
    py::array_t<Position> query_ends,
    const std::string& select,
    int num_threads = 1) {
       
    auto q_starts_ptr = static_cast<const Position*>(query_starts.request().ptr);
    auto q_ends_ptr = static_cast<const Position*>(query_ends.request().ptr);
    Index n_queries = query_starts.request().shape[0];

    bool quit_on_first = (select == "first");
    if (select != "all" && select != "first" && !quit_on_first) {
        throw std::runtime_error("Invalid 'select' parameter. Must be 'all' or 'first'.");
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
            std::vector<Index> nearest_matches;

            for (Index i = first, last = first + length; i < last; ++i) {
                nclist::NearestWorkspace<Index> ws_nearest;
                nclist::NearestParameters<Position> params;
                params.quit_on_first = quit_on_first;

                nclist::nearest(self.nclist_obj, q_starts_ptr[i], q_ends_ptr[i], params, ws_nearest, nearest_matches);

                if (!nearest_matches.empty()) {
                    if (select == "first" && !quit_on_first) {
                        all_results[i] = {nearest_matches.back()};
                    } else {
                        all_results[i] = nearest_matches;
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

pybind11::tuple perform_nearest(
    NCListSearchHandler &self,
    py::array_t<Position> query_starts,
    py::array_t<Position> query_ends,
    const std::string& select,
    int num_threads=1) {
    auto q_starts_ptr = static_cast<const Position*>(query_starts.request().ptr);
    auto q_ends_ptr = static_cast<const Position*>(query_ends.request().ptr);
    Index n_queries = query_starts.request().shape[0];


    bool quit_on_first = (select == "arbitrary");
    if (select != "all" && select != "arbitrary" && !quit_on_first) {
        throw std::runtime_error("Invalid 'select' parameter. Must be 'all' or 'arbitrary'.");
    }

    std::vector<std::vector<Index>> all_results(n_queries);
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
            std::vector<Index> nearest_matches;

            for (Index i = first, last = first + length; i < last; ++i) {
                nclist::NearestWorkspace<Index> ws_nearest;
                nclist::NearestParameters<Position> params;
                params.quit_on_first = quit_on_first;

                nclist::nearest(self.nclist_obj, q_starts_ptr[i], q_ends_ptr[i], params, ws_nearest, nearest_matches);

                if (!nearest_matches.empty()) {
                    if (select == "first" && !quit_on_first) {
                        all_results[i] = {nearest_matches.back()};
                    } else {
                        all_results[i] = nearest_matches;
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


void init_nclistsearch(pybind11::module &m){

    py::class_<NCListSearchHandler>(m, "NCListSearchHandler", "Manages nearest neighbor queries.")
        .def(py::init<py::array_t<Position>, py::array_t<Position>>(),
            py::arg("starts"), py::arg("ends"))

        .def("precede", &perform_precede,
            py::arg("query_starts"),
            py::arg("query_ends"),
            py::arg("select") = "first",
            py::arg("num_threads") = 1,
            "Find nearest positions that are downstream/follow each query range.")

        .def("follow", &perform_follow,
            py::arg("query_starts"),
            py::arg("query_ends"),
            py::arg("select") = "last",
            py::arg("num_threads") = 1,
            "Find nearest positions that are upstream/precede each query range.")

        .def("nearest", &perform_nearest,
            py::arg("query_starts"),
            py::arg("query_ends"),
            py::arg("select") = "arbitrary",
            py::arg("num_threads") = 1,
            "Find nearest ranges in both directions.");
}
