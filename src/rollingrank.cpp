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

template <class T>
py::array_t<double> rollingrank(py::array_t<T> x, int w, const char *method, bool pct) {
    const auto n = x.size();
    py::array_t<double> y(n);

    const auto compare = [&x](int a, int b) {
        return *x.data(a) < *x.data(b);
    };
    std::multiset<int, decltype(compare)> sorted_indices(compare);

    const auto rank_method = str_to_rank_method(method);

    for (int i = 0; i < n; i++) {
        if (i < w - 1) {
            sorted_indices.insert(i);
            *y.mutable_data(i) = std::numeric_limits<double>::quiet_NaN();
        } else {
            const auto iter = sorted_indices.insert(i);

            // debug
//            std::cout << "loop " << i << std::endl;
//            for (auto it = sorted_indices.begin(); it != sorted_indices.end(); ++it) {
//                std::cout << *it << " ";
//            }
//            std::cout << std::endl;

            double rank;

            switch (rank_method) {
                case RankMethod::Average:
                    {
                        const auto lower = sorted_indices.lower_bound(i);
                        const auto upper = sorted_indices.upper_bound(i);
                        rank = std::distance(sorted_indices.begin(), lower) + 0.5 * (std::distance(lower, upper) - 1);
                    }
                    break;
                case RankMethod::Min:
                    rank = std::distance(sorted_indices.begin(), sorted_indices.lower_bound(i));
                    break;
                case RankMethod::Max:
                    rank = std::distance(sorted_indices.begin(), sorted_indices.upper_bound(i)) - 1;
                    break;
                case RankMethod::First:
                    rank = std::distance(sorted_indices.begin(), iter);
                    break;
                default:
                    rank = std::numeric_limits<double>::quiet_NaN();
                    break;
            }

            if (pct) {
                // It uses division rather than reciprocal multiplication to ensure accuracy.
                // It may be slow, but the latency may hide in the latency of multiset processing.
                rank /= w - 1;
            }

            *y.mutable_data(i) = rank;
            sorted_indices.erase(sorted_indices.find(i - w + 1));
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

