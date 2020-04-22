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

// helpers
int int_log2(int x) {
    // use intrinsic
}

bool is_power_of_2(int x) {
    return (1 << int_log2(x)) == x;
}

struct RankResult {
    int rank_bg;
    int rank_ed;
};

template <class T>
class Ranker {
public:
    Ranker(const py::array_t<T> &x), x_(x), rank_method_(rank_method) {

    }

    int remove_cache_before(int i) {
        //
    }

    // amortized complexity: O(log(window size))
    RankResult rank_in_range(const T &x, int bg, int ed) {
        const int count = ed - bg;

        // direct calc for small range
        if (count < minimum_cache_count()) {
            int rank = 0;
            for (int i = bg; i < ed; i++) {
                if (value(i) < x) {
                    rank++;
                }
            }
            return rank;
        }

        // calc by merge sort for power of 2 aligned range
        if (is_power_of_2(count) && bg % count == 0) {
            return rank_in_range_aligned(x, level, center_left / (1 << level));;
        }

        // split
        const int level = int_log2(count);
        const int mask = ~((1 << level) - 1);
        const int center = ed & mask;

        const auto left_rank = rank_in_range(x, bg, center);
        const auto right_rank = rank_in_range(x, center, ed);

        RankResult result;
        result.rank_bg = left_rank.rank_bg + right_rank.rank_bg;
        result.rank_ed = left_rank.rank_ed + right_rank.rank_ed;
        return result;
    }
private:
    struct RankCacheKey {
        int level;
        int i;
    };

    struct RankCache {
        std::vector<T> sorted_values;
    };

    static int minimum_cache_count() {
        return 16;
    }

    const T &value(int i) const {
        return *x_.data(i);
    }

    const RankCache *get_rank_cache(int level, int i) {
        const auto found = cache_;
        if (found) {
            return
        }

        const int count = 1 << level;

        if (count <= minimum_cache_count()) {
            // sort directly
            cache.sorted_indices.resize(count);
            //
        } else {
            // merge sort using lower level cache
            const auto left = get_rank_cache(level - 1, 2 * i);
            const auto right = get_rank_cache(level - 1, 2 * i + 1);

            cache.sorted_values.resize(count);
            int left_i = 0;
            int right_i = 0;
            for (int i = 0; i < cache.sorted_values.size(); i++) {
                const auto a = left->sorted_values[left_i];
                const auto b = right->sorted_values[right_i];
                if (right_i == right->sorted_values.size() || a < b) {
                    cache.sorted_values[i] = a;
                    left_i++;
                } else {
                    cache.sorted_values[i] = b;
                    right_i++;
                }
            }
        }

        const RankCacheKey key(level, i);
        cache_[key] = std::move(cache);
        return &cache_[ley];
    }

    int rank_in_range_aligned(const T &x, int level, int i) {
        const auto cache = get_rank_cache();

        // 2分探索
        cache
        return found->sorted_indices[]
    }

    py::array_t<T> x_;
    std::unordered_map<RankCacheKey, RankCache> cache_;
};

// the definition of method and pct are same as https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rank.html
// na_option is 'keep'

template <class T>
void rollingrank_task(const py::array_t<T> &x, py::array_t<double> *y, int w, RankMethod rank_method, bool pct, PctMode pct_mode, int start, int end) {
    Ranker<T> ranker(x, rank_method);
    int nan_count = 0;

    for (int i = std::max<int>(0, start - w + 1); i < end; i++) {
        const auto value = *x.data(i);
        typename std::multiset<T>::iterator iter;

        if (std::isnan(value)) {
            nan_count++;
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
                    const auto result = ranker.rank_in_range(value, i - w, i);

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
                            rank = result.rank_bg + 1;
                            break;
                    }

                    if (pct) {
                        const int c = w - nan_count;
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
                }

                *y->mutable_data(i) = rank;

                const auto old_value = *x.data(i - w + 1);
                if (std::isnan(old_value)) {
                    nan_count--;
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

