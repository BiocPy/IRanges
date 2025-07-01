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

        self_starts.assign(starts_ptr, starts_ptr + n);
        self_ends.assign(ends_ptr, ends_ptr + n);

        sorted_starts.resize(n);
        sorted_ends.resize(n);
        for (Index i = 0; i < n; ++i) {
            sorted_starts[i] = {self_starts[i], i};
            sorted_ends[i] = {self_ends[i], i};
        }

        std::sort(sorted_starts.begin(), sorted_starts.end());
        std::sort(sorted_ends.begin(), sorted_ends.end());
    }

    nclist::Nclist<Index, Position> nclist_obj;

    std::vector<Position> self_starts;
    std::vector<Position> self_ends;
    std::vector<std::pair<Position, Index>> sorted_starts;
    std::vector<std::pair<Position, Index>> sorted_ends;
};

py::object perform_follow(
    NCListSearchHandler &self,
    py::array_t<Position> query_starts,
    const std::string& select,
    int num_threads = 1) {
    auto q_starts_ptr = static_cast<const Position*>(query_starts.request().ptr);
    Index n_queries = query_starts.request().shape[0];

    std::vector<Index> results(n_queries);
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
            for (Index i = first, last = first + length; i < last; ++i) {
                Position q_start = q_starts_ptr[i];

                // Binary search on sorted ends to find the last subject ending before the query starts
                auto it = std::upper_bound(self.sorted_ends.begin(), self.sorted_ends.end(), std::make_pair(q_start, Index(-1)));

                if (it == self.sorted_ends.begin()) {
                    results[i] = -1; // No preceding range
                } else {
                    --it;
                    results[i] = it->second;
                }
            }
        }, jobs_so_far, current_jobs);
        jobs_so_far += current_jobs;
    }

    for (auto& worker : workers) {
        worker.join();
    }

    if (select == "last") {
        return py::array_t<Index>(n_queries, results.data());
    } else {
        std::vector<Index> q_hits, s_hits;
        for (Index i = 0; i < n_queries; ++i) {
            if (results[i] != -1) {
                q_hits.push_back(i);
                s_hits.push_back(results[i]);
            }
        }
        return py::make_tuple(py::array_t<Index>(q_hits.size(), q_hits.data()), py::array_t<Index>(s_hits.size(), s_hits.data()));
    }
}

py::object perform_precede(
    NCListSearchHandler &self,
    py::array_t<Position> query_ends,
    const std::string& select,
    int num_threads = 1) {
    auto q_ends_ptr = static_cast<const Position*>(query_ends.request().ptr);
    Index n_queries = query_ends.request().shape[0];

    std::vector<Index> results(n_queries);
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
            for (Index i = first, last = first + length; i < last; ++i) {
                Position q_end = q_ends_ptr[i];

                // Binary search on sorted starts to find the first subject starting after the query ends
                auto it = std::lower_bound(self.sorted_starts.begin(), self.sorted_starts.end(), std::make_pair(q_end, Index(-1)));

                if (it == self.sorted_starts.end()) {
                    results[i] = -1; // No following range
                } else {
                    results[i] = it->second;
                }
            }
        }, jobs_so_far, current_jobs);
        jobs_so_far += current_jobs;
    }

    for (auto& worker : workers) {
        worker.join();
    }

    if (select == "first") {
        return py::array_t<Index>(n_queries, results.data());
    } else {
        std::vector<Index> q_hits, s_hits;
        for (Index i = 0; i < n_queries; ++i) {
            if (results[i] != -1) {
                q_hits.push_back(i);
                s_hits.push_back(results[i]);
            }
        }
        return py::make_tuple(py::array_t<Index>(q_hits.size(), q_hits.data()), py::array_t<Index>(s_hits.size(), s_hits.data()));
    }
}

py::object perform_nearest(
    NCListSearchHandler &self,
    py::array_t<Position> query_starts,
    py::array_t<Position> query_ends,
    const std::string& select,
    int num_threads=1) {
    auto q_starts_ptr = static_cast<const Position*>(query_starts.request().ptr);
    auto q_ends_ptr = static_cast<const Position*>(query_ends.request().ptr);
    Index n_queries = query_starts.request().shape[0];

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
            nclist::OverlapsAnyWorkspace<Index> ws_any;
            nclist::OverlapsAnyParameters<Position> ol_params;
            ol_params.min_overlap = 1;

            for (Index i = first, last = first + length; i < last; ++i) {
                Position q_start = q_starts_ptr[i];
                Position q_end = q_ends_ptr[i];

                std::vector<std::pair<Position, Index>> candidates;

                std::vector<Index> overlaps;
                nclist::overlaps_any(self.nclist_obj, q_start, q_end, ol_params, ws_any, overlaps);
                for (const auto& ol_idx : overlaps) {
                    candidates.emplace_back(0, ol_idx);
                }

                auto it_b = std::upper_bound(self.sorted_ends.begin(), self.sorted_ends.end(), std::make_pair(q_start, std::numeric_limits<Index>::max()));
                if (it_b != self.sorted_ends.begin()) {
                    --it_b;
                    candidates.emplace_back(q_start - it_b->first, it_b->second);
                }

                auto it_a = std::lower_bound(self.sorted_starts.begin(), self.sorted_starts.end(), std::make_pair(q_end, Index(-1)));
                if (it_a != self.sorted_starts.end()) {
                    candidates.emplace_back(it_a->first - q_end, it_a->second);
                }

                if (candidates.empty()) {
                    continue;
                }

                Position min_dist = candidates[0].first;
                for (size_t k = 1; k < candidates.size(); ++k) {
                    if (candidates[k].first < min_dist) {
                        min_dist = candidates[k].first;
                    }
                }

                std::vector<Index> best_hits;
                for (const auto& cand : candidates) {
                    if (cand.first == min_dist) {
                        best_hits.push_back(cand.second);
                    }
                }

                if (select == "arbitrary") {
                    std::sort(best_hits.begin(), best_hits.end());
                    all_results[i] = {best_hits[0]};
                } else {
                    all_results[i] = best_hits;
                }
            }
        }, jobs_so_far, current_jobs);
        jobs_so_far += current_jobs;
    }

    for (auto& worker : workers) {
        worker.join();
    }

    if (select == "arbitrary") {
        py::array_t<Index> result(n_queries);
        auto res_ptr = static_cast<Index*>(result.request().ptr);
        for (Index i = 0; i < n_queries; ++i) {
            res_ptr[i] = all_results[i].empty() ? -1 : all_results[i][0];
        }
        return std::move(result);
    } else {
        std::vector<std::pair<Index, Index>> final_pairs;
        for (Index i = 0; i < n_queries; ++i) {
            for (const auto& self_hit : all_results[i]) {
                final_pairs.emplace_back(i, self_hit);
            }
        }

        std::stable_sort(final_pairs.begin(), final_pairs.end());

        py::array_t<Index> query_hits(final_pairs.size());
        py::array_t<Index> self_hits(final_pairs.size());
        auto q_res_ptr = static_cast<Index*>(query_hits.request().ptr);
        auto s_res_ptr = static_cast<Index*>(self_hits.request().ptr);

        for(size_t i = 0; i < final_pairs.size(); ++i) {
            q_res_ptr[i] = final_pairs[i].first;
            s_res_ptr[i] = final_pairs[i].second;
        }
        return py::make_tuple(query_hits, self_hits);
    }
}


void init_nclistsearch(pybind11::module &m){

    py::class_<NCListSearchHandler>(m, "NCListSearchHandler", "Manages nearest neighbor queries.")
        .def(py::init<py::array_t<Position>, py::array_t<Position>>(),
            py::arg("starts"), py::arg("ends"))

        .def("precede", &perform_precede,
            py::arg("query_ends"),
            py::arg("select") = "first",
            py::arg("num_threads") = 1,
            "Find nearest positions that are downstream/follow each query range.")

        .def("follow", &perform_follow,
            py::arg("query_starts"),
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
