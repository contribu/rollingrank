#include <cstdint>
#include <algorithm>
#include <limits>
#include <set>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace {
namespace py = pybind11;

template <class T>
py::array_t<double> rollingrank(py::array_t<T> x, int w, bool pct) {
    const auto n = x.size();
    py::array_t<double> y(n);

    const auto compare = [&x](int a, int b) {
        return *x.data(a) < *x.data(b);
    };
    std::multiset<int, decltype(compare)> sorted_indices(compare);

    for (int i = 0; i < n; i++) {
        if (i < w - 1) {
            sorted_indices.insert(i);
            *y.mutable_data(i) = std::numeric_limits<double>::quiet_NaN();
        } else {
            const auto iter = sorted_indices.insert(i);
            double rank = std::distance(sorted_indices.begin(), iter);
            if (pct) {
                // It uses division rather than reciprocal multiplication to ensure accuracy.
                // It may be slow, but the latency may hide in the latency of multiset processing.
                rank /= w - 1;
            }
            *y.mutable_data(i) = rank;
            sorted_indices.erase(i - w + 1);
        }
    }

    return y;
}
}

#define def_rollingrank(type) \
    m.def("rollingrank", &rollingrank<type>, "", py::arg("x"), py::arg("window"), py::arg("pct") = false);

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

