#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include <set>
#include <thread>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <taskflow/taskflow.hpp>

//#include <iostream>

namespace py = pybind11;

namespace {

enum class RankMethod {
    Average, Min, Max, First
};

enum class PctMode {
    Pandas, Closed
};

RankMethod str_to_rank_method(const char *method) {
    if (std::strcmp(method, "average") == 0) {
        return RankMethod::Average;
    }
    else if (std::strcmp(method, "min") == 0) {
        return RankMethod::Min;
    }
    else if (std::strcmp(method, "max") == 0) {
        return RankMethod::Max;
    }
    else if (std::strcmp(method, "first") == 0) {
        return RankMethod::First;
    }
    else {
        return RankMethod::Average;
    }
}

PctMode str_to_pct_mode(const char *str) {
    if (std::strcmp(str, "pandas") == 0) {
        return PctMode::Pandas;
    }
    else if (std::strcmp(str, "closed") == 0) {
        return PctMode::Closed;
    }
    else {
        return PctMode::Pandas;
    }
}

// the definition of method and pct are same as https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rank.html
// na_option is 'keep'

template <class T>
void rollingrank_task(const py::array_t<T> &x, py::array_t<double> *y, int w, RankMethod rank_method, bool pct, PctMode pct_mode, int start, int end) {
    std::multiset<T> sorted_indices;
    int nan_count = 0;

    for (int i = std::max<int>(0, start - w); i < end; i++) {
        const auto value = *x.data(i);
        typename std::multiset<T>::iterator iter;

        if (std::isnan(value)) {
            nan_count++;
        }
        else {
            iter = sorted_indices.insert(value);
        }

        if (start <= i) {
            if (nan_count + sorted_indices.size() < w) {
                *y->mutable_data(i) = std::numeric_limits<double>::quiet_NaN();
            }
            else {
                double rank;

                if (std::isnan(value)) {
                    rank = std::numeric_limits<double>::quiet_NaN();
                }
                else {
                    switch (rank_method) {
                        case RankMethod::Average:
                            {
                                const auto range = sorted_indices.equal_range(value);
                                rank = std::distance(sorted_indices.begin(), range.first) + 0.5 * (std::distance(range.first, range.second) - 1) + 1;
                            }
                            break;
                        case RankMethod::Min:
                            rank = std::distance(sorted_indices.begin(), sorted_indices.lower_bound(value)) + 1;
                            break;
                        case RankMethod::Max:
                            rank = std::distance(sorted_indices.begin(), sorted_indices.upper_bound(value));
                            break;
                        case RankMethod::First:
                            rank = std::distance(sorted_indices.begin(), iter) + 1;
                            break;
                        default:
                            rank = std::numeric_limits<double>::quiet_NaN();
                            break;
                    }

                    if (pct) {
                        switch (pct_mode) {
                            case PctMode::Pandas:
                                rank /= sorted_indices.size();
                                break;
                            case PctMode::Closed:
                                if (sorted_indices.size() == 1) {
                                    rank = 0.5;
                                } else {
                                    rank = (rank - 1) / (sorted_indices.size() - 1);
                                }
                        }
                    }
                }

                *y->mutable_data(i) = rank;

                const auto old_value = *x.data(i - w + 1);
                if (std::isnan(old_value)) {
                    nan_count--;
                }
                else {
                    sorted_indices.erase(sorted_indices.find(old_value));
                }
            }
        }
    }
}

template <class T>
py::array_t<double> rollingrank(py::array_t<T> x, int w, const char *method, bool pct, const char *_pct_mode, int n_jobs) {
    const auto n = x.size();
    py::array_t<double> y(n);

    const auto rank_method = str_to_rank_method(method);
    const auto pct_mode = str_to_pct_mode(_pct_mode);

    const auto split_size = std::max<int>(w, 10000);
    const auto groups = (n + split_size - 1) / split_size;

    if (groups < 2) {
        rollingrank_task<T>(x, &y, w, rank_method, pct, pct_mode, 0, n);
    }
    else {
        const auto split_size2 = (n + groups - 1) / groups;
        tf::Taskflow taskflow;

        for (int i = 0; i < groups; i++) {
            const auto start = i * split_size2;
            const auto end = std::min<int>(n, (i + 1) * split_size2);

            taskflow.emplace([start, end, x, &y, w, rank_method, pct, pct_mode]() {
                rollingrank_task<T>(x, &y, w, rank_method, pct, pct_mode, start, end);
            });
        }

        tf::Executor executor(n_jobs < 0 ? std::thread::hardware_concurrency() : n_jobs);
        executor.run(taskflow);
        executor.wait_for_all();
    }

    return y;
}
}

#define def_rollingrank(type) \
    m.def("rollingrank", &rollingrank<type>, "", \
        py::arg("x"), \
        py::arg("window"), \
        py::arg("method") = "average", \
        py::arg("pct") = false, \
        py::arg("pct_mode") = "pandas", \
        py::arg("n_jobs") = -1 \
    );

PYBIND11_MODULE(rollingrank_native, m) {
    m.doc() = "rolling rank for numpy array";

    def_rollingrank(int8_t)
    def_rollingrank(int16_t)
    def_rollingrank(int32_t)
    def_rollingrank(int64_t)
    def_rollingrank(uint8_t)
    def_rollingrank(uint16_t)
    def_rollingrank(uint32_t)
    def_rollingrank(uint64_t)
    def_rollingrank(bool)
    def_rollingrank(float)
    def_rollingrank(double)
}

