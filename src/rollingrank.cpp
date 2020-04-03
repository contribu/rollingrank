#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include <set>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

//#include <iostream>

namespace {
namespace py = pybind11;

enum class RankMethod {
    Average, Min, Max, First
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

// the definition of method and pct are same as https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rank.html
// na_option is 'keep'

template <class T>
py::array_t<double> rollingrank(py::array_t<T> x, int w, const char *method, bool pct) {
    const auto n = x.size();
    py::array_t<double> y(n);

    std::multiset<T> sorted_indices;

    const auto rank_method = str_to_rank_method(method);

    for (int i = 0; i < n; i++) {
        const auto value = *x.data(i);

        if (i < w - 1) {
            if (!std::isnan(value)) {
                sorted_indices.insert(value);
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
                const auto iter = sorted_indices.insert(value);

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
                    // It uses division rather than reciprocal multiplication to ensure accuracy.
                    // It may be slow, but the latency may hide in the latency of multiset processing.
                    rank /= sorted_indices.size();
                }
            }

            *y.mutable_data(i) = rank;

            const auto old_value = *x.data(i - w + 1);
            if (!std::isnan(old_value)) {
                sorted_indices.erase(sorted_indices.find(old_value));
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
        py::arg("pct") = false \
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

