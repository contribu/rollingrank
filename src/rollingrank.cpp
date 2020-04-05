#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include <set>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "avl_array.h"

//#include <iostream>

namespace {
namespace py = pybind11;

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
py::array_t<double> rollingrank(py::array_t<T> x, size_t w, const char *method, bool pct, const char *_pct_mode) {
    const size_t n = x.size();
    py::array_t<double> y(n);

    typedef std::pair<double, size_t> Pair;
    avl_array<Pair, bool, size_t, 2048, true> avl;

    const auto rank_method = str_to_rank_method(method);
    const auto pct_mode = str_to_pct_mode(_pct_mode);

    for (size_t i = 0; i < n; i++) {
        const auto value = *x.data(i);

        if (i < w - 1) {
            if (!std::isnan(value)) {
                avl.insert(Pair(value, i), true);
            }
            *y.mutable_data(i) = std::numeric_limits<double>::quiet_NaN();
        } else {
            // debug
//            std::cout << "loop " << i << std::endl;
//            for (auto it = sorted_indices.begin(); it != sorted_indices.end(); ++it) {
//                std::cout << *it << " ";
//            }
//            std::cout << std::endl;

            double rank;

            if (std::isnan(value)) {
                rank = std::numeric_limits<double>::quiet_NaN();
            }
            else {
                avl.insert(Pair(value, i), true);

                switch (rank_method) {
                    case RankMethod::Average:
                        {
                            const auto min_rank = avl.lower_bound_rank(Pair(value, 0)) + 1;
                            const auto max_rank = avl.lower_bound_rank(Pair(value, std::numeric_limits<size_t>::max())) - 1 + 1;
                            rank = 0.5 * (min_rank + max_rank);
                        }
                        break;
                    case RankMethod::Min:
                        rank = avl.lower_bound_rank(Pair(value, 0)) + 1;
                        break;
                    case RankMethod::Max:
                        rank = avl.lower_bound_rank(Pair(value, std::numeric_limits<size_t>::max())) - 1 + 1;
                        break;
                    case RankMethod::First:
                        rank = avl.lower_bound_rank(Pair(value, i)) + 1;
                        break;
                    default:
                        rank = std::numeric_limits<double>::quiet_NaN();
                        break;
                }

                if (pct) {
                    switch (pct_mode) {
                        case PctMode::Pandas:
                            rank /= avl.size();
                            break;
                        case PctMode::Closed:
                            if (avl.size() == 1) {
                                rank = 0.5;
                            } else {
                                rank = (rank - 1) / (avl.size() - 1);
                            }
                    }
                }
            }

            *y.mutable_data(i) = rank;

            const auto old_value = *x.data(i - w + 1);
            if (!std::isnan(old_value)) {
                avl.erase(Pair(old_value, i - w + 1));
            }
        }
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
        py::arg("pct_mode") = "pandas" \
    );

PYBIND11_MODULE(rollingrank, m) {
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

