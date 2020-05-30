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
#include <rank_in_range/rank_in_range.h>

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

template <class T>
class NumpyIterator {
public:
    NumpyIterator(const pybind11::detail::unchecked_reference<T, 1> *x, int base = 0): x_(x), base_(base) {}

    const T &operator [](int i) const {
        return (*x_)(base_ + i);
    }

    const T &operator *() const {
        return (*x_)(base_);
    }

    NumpyIterator<T> operator +(int x) const {
        return NumpyIterator<T>(x_, base_ + x);
    }

    NumpyIterator<T> &operator ++() {
        base_++;
        return *this;
    }

    bool operator !=(const NumpyIterator<T> &other) const {
        return base_ != other.base_ || x_ != other.x_;
    }
private:
    const pybind11::detail::unchecked_reference<T, 1> *x_;
    int base_;
};

// the definition of method and pct are same as https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rank.html
// na_option is 'keep'

template <class T>
void rollingrank_task(const py::array_t<T> &x, py::array_t<double> *y, int w, RankMethod rank_method, bool pct, PctMode pct_mode, int start, int end) {
    auto unchecked_x = x.template unchecked<1>();
    NumpyIterator<T> x_iter(&unchecked_x);
    rank_in_range::Ranker<T, NumpyIterator<T>> ranker(x_iter);

    for (int i = start; i < end; i++) {
        const auto value = *x.data(i);
        typename std::multiset<T>::iterator iter;

        if (i % w == 0) {
            ranker.remove_cache_before(i - w + 1);
        }

        if (i - w + 1 < 0 || std::isnan(value)) {
            *y->mutable_data(i) = std::numeric_limits<double>::quiet_NaN();
        }
        else {
            const auto result = ranker.rank_in_range(value, i - w + 1, i + 1);

            double rank;
            switch (rank_method) {
                case RankMethod::Average:
                    rank = 0.5 * (result.rank_bg + result.rank_ed - 1) + 1;
                    break;
                case RankMethod::Min:
                    rank = result.rank_bg + 1;
                    break;
                case RankMethod::Max:
                    rank = result.rank_ed;
                    break;
                case RankMethod::First:
                    rank = result.rank_ed;
                    break;
            }

            if (pct) {
                const int c = w - result.nan_count;
                switch (pct_mode) {
                    case PctMode::Pandas:
                        rank /= c;
                        break;
                    case PctMode::Closed:
                        if (c == 1) {
                            rank = 0.5;
                        } else {
                            rank = (rank - 1) / (c - 1);
                        }
                }
            }

            *y->mutable_data(i) = rank;
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

    if (n_jobs < 0) {
        n_jobs = std::thread::hardware_concurrency();
    }

    if (groups < 2 || n_jobs == 1) {
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

        tf::Executor executor(n_jobs);
        executor.run(taskflow);
        executor.wait_for_all();
    }

    return y;
}

// https://stackoverflow.com/questions/15843525/how-do-you-insert-the-value-in-a-sorted-vector
template <class T, class Compare>
void sorted_insert(std::vector<T> *sorted, T value, const Compare &compare) {
    sorted->insert(
        std::upper_bound(sorted->begin(), sorted->end(), value, compare),
        value
    );
}

// https://stackoverflow.com/questions/26719144/how-to-erase-a-value-efficiently-from-a-sorted-vector
template <class T, class Compare>
void sorted_erase(std::vector<T> *sorted, T value, const Compare &compare) {
    const auto lb = std::lower_bound(sorted->begin(), sorted->end(), value, compare);
    sorted->erase(lb);
}

template <class T>
void rci_task(const py::array_t<T> &x, py::array_t<double> *y, int w, int start, int end) {
    auto unchecked_x = x.template unchecked<1>();
    NumpyIterator<T> x_iter(&unchecked_x);
    std::vector<int> sorted;
    std::vector<int> ranks(w);
    int nan_count = 0;

    for (int i = std::max(0, start - w + 1); i < start; i++) {
        const auto value = *x.data(i);
        if (std::isnan(value)) {
            nan_count++;
        } else {
            sorted.push_back(i);
        }
    }
    const auto compare_func = [&x](int i, int j) {
        return *x.data(i) < *x.data(j);
    };
    std::sort(sorted.begin(), sorted.end(), compare_func);

    for (int i = start; i < end; i++) {
        const auto value = *x.data(i);
        typename std::multiset<T>::iterator iter;

        if (std::isnan(value)) {
            nan_count++;
        } else {
            sorted_insert(&sorted, i, compare_func);
        }

        if (sorted.size() + nan_count < w || std::isnan(value)) {
            *y->mutable_data(i) = std::numeric_limits<double>::quiet_NaN();
        }
        else {
            // rank method is average
            int prev_j = -1;
            for (int j = 0; j < sorted.size(); j++) {
                if (j + 1 == sorted.size() || *x.data(sorted[j]) != *x.data(sorted[j + 1])) {
                    for (int k = prev_j + 1; k <= j; k++) {
                        ranks[j] = (prev_j + 1 + j) * 0.5;
                    }
                    prev_j = j;
                }
            }

            double mean_t = 0;
            double mean_rank = 0;
            for (int j = 0; j < sorted.size(); j++) {
                mean_t += sorted[j];
                mean_rank += ranks[j];
            }
            mean_t /= sorted.size();
            mean_rank /= sorted.size();

            double var_t = 0;
            double var_rank = 0;
            double cov = 0;
            for (int j = 0; j < sorted.size(); j++) {
                const double t = sorted[j] - mean_t;
                const double rank = ranks[j] - mean_rank;
                var_t += t * t;
                var_rank += rank * rank;
                cov += t * rank;
            }

            *y->mutable_data(i) = cov / (1e-37 + std::sqrt(var_t * var_rank));
        }

        if (i - w + 1 >= 0) {
            const auto old_value = *x.data(i - w + 1);
            if (std::isnan(*x.data(old_value))) {
                nan_count--;
            } else {
                sorted_erase(&sorted, i - w + 1, compare_func);
            }
        }
    }
}

template <class T>
py::array_t<double> rci(py::array_t<T> x, int w, int n_jobs) {
    const auto n = x.size();
    py::array_t<double> y(n);

    const auto split_size = std::max<int>(w, 10000);
    const auto groups = (n + split_size - 1) / split_size;

    if (n_jobs < 0) {
        n_jobs = std::thread::hardware_concurrency();
    }

    if (groups < 2 || n_jobs == 1) {
        rci_task<T>(x, &y, w, 0, n);
    }
    else {
        const auto split_size2 = (n + groups - 1) / groups;
        tf::Taskflow taskflow;

        for (int i = 0; i < groups; i++) {
            const auto start = i * split_size2;
            const auto end = std::min<int>(n, (i + 1) * split_size2);

            taskflow.emplace([start, end, x, &y, w]() {
                rci_task<T>(x, &y, w, start, end);
            });
        }

        tf::Executor executor(n_jobs);
        executor.run(taskflow);
        executor.wait_for_all();
    }

    return y;
}
}

#define def_rci(type) \
    m.def("rci", &rci<type>, "", \
        py::arg("x"), \
        py::arg("window"), \
        py::arg("n_jobs") = -1 \
    );

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

    def_rci(int8_t)
    def_rci(int16_t)
    def_rci(int32_t)
    def_rci(int64_t)
    def_rci(uint8_t)
    def_rci(uint16_t)
    def_rci(uint32_t)
    def_rci(uint64_t)
    def_rci(bool)
    def_rci(float)
    def_rci(double)

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

