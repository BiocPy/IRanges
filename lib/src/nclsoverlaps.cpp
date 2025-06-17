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

namespace py = pybind11;

// probably not good to specify type ahead, but works for now
using Index = int;
using Position = int;
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
    int num_workers=1) {

    py::buffer_info q_starts_buf = query_starts.request();
    py::buffer_info q_ends_buf = query_ends.request();

    auto q_starts_ptr = static_cast<const Position*>(q_starts_buf.ptr);
    auto q_ends_ptr = static_cast<const Position*>(q_ends_buf.ptr);
    Index n_queries = q_starts_buf.shape[0];

    bool quit_on_first = (select == "first" || select == "arbitrary");
    if (select != "all" && select != "last" && !quit_on_first) {
        throw std::runtime_error("Invalid 'select' parameter. Must be 'all', 'first', 'last', or 'arbitrary'.");
    }

    std::vector<std::vector<Index> > final_results(n_queries);

    bool use_quit_on_first_opt = quit_on_first;

    if (query_type == "any") {
        nclist::OverlapsAnyParameters<Position> params;
        params.min_overlap = min_overlap > 0 ? min_overlap : 0;
        if (max_gap >= 0) {
            params.max_gap = max_gap;
        }
        params.quit_on_first = use_quit_on_first_opt;

        int num_jobs = n_queries / num_workers;
        int num_remaining = n_queries % num_workers;
        int jobs_so_far = 0;
        std::vector<std::thread> workers;
        workers.reserve(num_workers);
        for (int i = 0; i < num_workers; ++i) {
            int current_jobs = num_jobs + (i < num_remaining);
            
            if (current_jobs == 0) {
                break;
            }

            workers.emplace_back([&](int first, int length) -> void {
                std::vector<Index> single_query_matches;
                nclist::OverlapsAnyWorkspace<Index> ws_any;

                for (Index i = first, last = first + length; i < last; ++i) {
                // for (Index i = 0; i < n_queries; ++i) {
                    nclist::overlaps_any(self.nclist_obj, q_starts_ptr[i], q_ends_ptr[i], params, ws_any, single_query_matches);
                    if (!single_query_matches.empty()) {
                        if (select == "last") {
                            std::vector<Index> last_value = {single_query_matches.back()};
                            final_results[i] = last_value;
                        } else {
                            final_results[i] = single_query_matches;
                        }
                    }
                }
            }, jobs_so_far, current_jobs);

            jobs_so_far += current_jobs;
        }

        for (auto& worker : workers) {
            worker.join();
        }
    } 
    else if (query_type == "start") {
        nclist::OverlapsStartParameters<Position> params;
        params.min_overlap = min_overlap > 0 ? min_overlap : 0;
        if (max_gap >= 0) {
            params.max_gap = max_gap;
        }
        params.quit_on_first = use_quit_on_first_opt;

        int num_jobs = n_queries / num_workers;
        int num_remaining = n_queries % num_workers;
        int jobs_so_far = 0;
        std::vector<std::thread> workers;
        workers.reserve(num_workers);
        for (int i = 0; i < num_workers; ++i) {
            int current_jobs = num_jobs + (i < num_remaining);
            
            if (current_jobs == 0) {
                break;
            }

            workers.emplace_back([&](int first, int length) -> void {
                std::vector<Index> single_query_matches;
                nclist::OverlapsStartWorkspace<Index> ws_start;

                for (Index i = first, last = first + length; i < last; ++i) {
                    nclist::overlaps_start(self.nclist_obj, q_starts_ptr[i], q_ends_ptr[i], params, ws_start, single_query_matches);
                    if (!single_query_matches.empty()) {
                        if (select == "last") {
                            std::vector<Index> last_value = {single_query_matches.back()};
                            final_results[i] = last_value;
                        } else {
                            final_results[i] = single_query_matches;
                        }
                    }
                }
            }, jobs_so_far, current_jobs);

            jobs_so_far += current_jobs;
        }

        for (auto& worker : workers) {
            worker.join();
        }
    } else if (query_type == "end") {
        nclist::OverlapsEndParameters<Position> params;
        params.min_overlap = min_overlap > 0 ? min_overlap : 0;
        if (max_gap >= 0) {
            params.max_gap = max_gap;
        }
        params.quit_on_first = use_quit_on_first_opt;

        int num_jobs = n_queries / num_workers;
        int num_remaining = n_queries % num_workers;
        int jobs_so_far = 0;
        std::vector<std::thread> workers;
        workers.reserve(num_workers);
        for (int i = 0; i < num_workers; ++i) {
            int current_jobs = num_jobs + (i < num_remaining);
            
            if (current_jobs == 0) {
                break;
            }

            workers.emplace_back([&](int first, int length) -> void {
                std::vector<Index> single_query_matches;
                nclist::OverlapsEndWorkspace<Index> ws_end;

                for (Index i = first, last = first + length; i < last; ++i) {
                    nclist::overlaps_end(self.nclist_obj, q_starts_ptr[i], q_ends_ptr[i], params, ws_end, single_query_matches);
                    if (!single_query_matches.empty()) {
                        if (select == "last") {
                            std::vector<Index> last_value = {single_query_matches.back()};
                            final_results[i] = last_value;
                        } else {
                            final_results[i] = single_query_matches;
                        }
                    }
                }
            }, jobs_so_far, current_jobs);

            jobs_so_far += current_jobs;
        }

        for (auto& worker : workers) {
            worker.join();
        }
    } else if (query_type == "within") {
        nclist::OverlapsWithinParameters<Position> params;
        params.min_overlap = min_overlap > 0 ? min_overlap : 0;
        if (max_gap >= 0) {
            params.max_gap = max_gap;
        }
        params.quit_on_first = use_quit_on_first_opt;

        int num_jobs = n_queries / num_workers;
        int num_remaining = n_queries % num_workers;
        int jobs_so_far = 0;
        std::vector<std::thread> workers;
        workers.reserve(num_workers);
        for (int i = 0; i < num_workers; ++i) {
            int current_jobs = num_jobs + (i < num_remaining);
            
            if (current_jobs == 0) {
                break;
            }

            workers.emplace_back([&](int first, int length) -> void {
                std::vector<Index> single_query_matches;
                nclist::OverlapsWithinWorkspace<Index> ws_within;

                for (Index i = first, last = first + length; i < last; ++i) {
                    nclist::overlaps_within(self.nclist_obj, q_starts_ptr[i], q_ends_ptr[i], params, ws_within, single_query_matches);
                    if (!single_query_matches.empty()) {
                        if (select == "last") {
                            std::vector<Index> last_value = {single_query_matches.back()};
                            final_results[i] = last_value;
                        } else {
                            final_results[i] = single_query_matches;
                        }
                    }
                }
            }, jobs_so_far, current_jobs);

            jobs_so_far += current_jobs;
        }

        for (auto& worker : workers) {
            worker.join();
        }
    } else {
        throw std::runtime_error("Invalid query_type. Must be 'any', 'start', 'end', or 'within'.");
    }

    std::size_t counter = 0;
    for (const auto &res:final_results) {
        counter += res.size();
    }
    pybind11::array_t<Index> self_hits(counter);
    pybind11::array_t<Index> query_hits(counter);

    py::buffer_info s_res_buf = self_hits.request();
    py::buffer_info q_res_buf = query_hits.request();
    auto s_res_ptr = static_cast<Index*>(s_res_buf.ptr);
    auto q_res_ptr = static_cast<Index*>(q_res_buf.ptr);

    counter = 0;
    int index = 0;
    for (const auto &res:final_results) {
        std::copy(res.begin(), res.end(), s_res_ptr + counter);
        std::fill_n(q_res_ptr + counter, res.size(), index);
        index += 1;
        counter += res.size();
    }

    pybind11::tuple output(2);
    output[0] = self_hits;
    output[1] = query_hits;

    return output;
}


void init_nclist(pybind11::module &m){

    py::class_<NCListHandler>(m, "NCListHandler", "Manages an nclist-cpp index and workspaces for overlap queries.")
        .def(py::init<py::array_t<Position>, py::array_t<Position>>(),
             py::arg("starts"), py::arg("ends"),
             "Builds the NClist index from subject intervals.\n\n"
             "Args:\n"
             "    starts (np.ndarray): A numpy array of start positions.\n"
             "    ends (np.ndarray): A numpy array of end positions.")

        .def("find_overlaps",
             &perform_find_overlaps,
             py::arg("query_starts"),
             py::arg("query_ends"),
             py::arg("query_type") = "any",
             py::arg("select") = "all",
             py::arg("max_gap") = -1,
             py::arg("min_overlap") = 1,
             py::arg("num_workers") = 1,
R"doc(Finds overlaps between query intervals and the indexed (self) subject intervals.

Args:
    query_starts (np.ndarray): A numpy array of start positions for the queries.
    query_ends (np.ndarray): A numpy array of end positions for the queries.
    query_type (str, optional): The type of overlap to perform.
        Defaults to "any".
        - "any": Any overlap is reported.
        - "start": The query must lie inside the subject.
        - "end": The subject must lie inside the query.
        - "within": Same as "start".
    select (str, optional): Which overlapping pairs to report.
        Defaults to "all".
        - "all": Reports all pairs.
        - "first": For each query, reports the first overlapping subject.
        - "last": For each query, reports the last overlapping subject.
        - "arbitrary": For each query, reports an arbitrary overlapping subject (implemented as "first").
    max_gap (int, optional): The maximum gap allowed between intervals for them
        to be considered overlapping. A negative value means no gap is allowed.
        Defaults to -1.
    min_overlap (int, optional): The minimum number of positions that must
        overlap for a pair to be reported. Defaults to 1.
    num_workers (int, optional): Number of threads to use. Defaults to 1.

Returns:
    tuple[list[int], list[int]]]: A list of (query_index, subject_index) pairs
    representing the found overlaps.
)doc");
}
