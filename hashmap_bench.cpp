#include <benchmark/benchmark.h>

#include <atomic>
#include <random>
#include <unordered_map>

#include "hashmap.hpp"

// ===========================================================================
// SINGLE-THREADED BENCHMARKS
//
// concurrent explicitly set to FALSE here -- this measures the plain
// non-atomic code path (no CAS, no acquire/release), which is the fair
// comparison against std::unordered_map. The concurrent=true numbers
// belong in the concurrent benchmarks below, even when run with one
// thread, since that path still pays atomic-op costs the non-concurrent
// path never does.
// ===========================================================================

static void BM_StdHashMap_Insert(benchmark::State& state) {
    std::mt19937 rng(42);
    for (auto _ : state) {
        std::unordered_map<int, int> map;
        for (int i = 0; i < state.range(0); ++i) {
            map.insert({rng(), i});
        }
        benchmark::DoNotOptimize(map);
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_StdHashMap_Insert)->Range(64, 1 << 20)->Complexity();

static void BM_StdHashMap_Lookup(benchmark::State& state) {
    std::unordered_map<int, int> map;
    std::mt19937 rng(42);
    std::vector<int> keys;
    for (int i = 0; i < state.range(0); ++i) {
        int key = rng();
        map.insert({key, i});
        keys.push_back(key);
    }
    size_t idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(map.at(keys[idx++ % keys.size()]));
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_StdHashMap_Lookup)->Range(64, 1 << 20)->Complexity();

// ---- Templated so the same benchmark body can be instantiated once per
// load-factor configuration below, instead of copy-pasting per ratio.
template <typename MapT>
static void BM_Insert_Templated(benchmark::State& state) {
    std::mt19937 rng(42);
    for (auto _ : state) {
        MapT map;
        for (int i = 0; i < state.range(0); ++i) {
            map.insert(rng(), i);
        }
        benchmark::DoNotOptimize(map);
    }
    state.SetComplexityN(state.range(0));
}

template <typename MapT>
static void BM_Lookup_Templated(benchmark::State& state) {
    MapT map;
    std::mt19937 rng(42);
    std::vector<int> keys;
    for (int i = 0; i < state.range(0); ++i) {
        int key = rng();
        map.insert(key, i);
        keys.push_back(key);
    }
    size_t idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(map.at(keys[idx++ % keys.size()]));
    }
    state.SetComplexityN(state.range(0));
}

// Default ratio (matches the class's own default -- 7/8, i.e. resize at
// 87.5% full) plus two comparison points: a looser 1/2 (resizes early,
// more memory, shorter probe chains) and a tighter 9/10 (resizes late,
// less memory, longer probe chains -- stress-tests probing under a
// nearly-full table).
using DefaultLoadMap = hm::HashMap<int, int, /*concurrent=*/false>;
using LooseLoadMap = hm::HashMap<int, int, /*concurrent=*/false, 1, 2>;
using TightLoadMap = hm::HashMap<int, int, /*concurrent=*/false, 9, 10>;

BENCHMARK_TEMPLATE(BM_Insert_Templated, DefaultLoadMap)
    ->Range(64, 1 << 20)
    ->Complexity()
    ->Name("BM_YourHashMap_Insert_LoadFactor_7_8_default");
BENCHMARK_TEMPLATE(BM_Insert_Templated, LooseLoadMap)
    ->Range(64, 1 << 20)
    ->Complexity()
    ->Name("BM_YourHashMap_Insert_LoadFactor_1_2_loose");
BENCHMARK_TEMPLATE(BM_Insert_Templated, TightLoadMap)
    ->Range(64, 1 << 20)
    ->Complexity()
    ->Name("BM_YourHashMap_Insert_LoadFactor_9_10_tight");

BENCHMARK_TEMPLATE(BM_Lookup_Templated, DefaultLoadMap)
    ->Range(64, 1 << 20)
    ->Complexity()
    ->Name("BM_YourHashMap_Lookup_LoadFactor_7_8_default");
BENCHMARK_TEMPLATE(BM_Lookup_Templated, LooseLoadMap)
    ->Range(64, 1 << 20)
    ->Complexity()
    ->Name("BM_YourHashMap_Lookup_LoadFactor_1_2_loose");
BENCHMARK_TEMPLATE(BM_Lookup_Templated, TightLoadMap)
    ->Range(64, 1 << 20)
    ->Complexity()
    ->Name("BM_YourHashMap_Lookup_LoadFactor_9_10_tight");

// ===========================================================================
// CONCURRENT BENCHMARKS
//
// Google Benchmark's ->Threads(N) runs N copies of the benchmark BODY
// concurrently, but each thread gets its own State -- to share ONE map
// across all of them, we use the documented idiom: a Fixture whose
// SetUp/TearDown are only really executed by thread_index == 0, relying
// on Google Benchmark's guarantee that ALL threads' SetUp calls complete
// (a barrier) before ANY thread's benchmark body runs, and that all
// bodies finish (another barrier) before TearDown runs. This is the same
// pattern used in Google Benchmark's own multithreaded examples.
// ===========================================================================

using ConcurrentMap = hm::HashMap<int, int, /*concurrent=*/true>;

class ConcurrentMapFixture : public benchmark::Fixture {
  public:
    void SetUp(const benchmark::State& state) override {
        if (state.thread_index() == 0) {
            map_ = new ConcurrentMap();
            // Pre-populate so lookup/mixed benchmarks have real data to
            // find -- range(0) is the total key count, split evenly
            // across whatever thread count this run uses.
            int total_keys = static_cast<int>(state.range(0));
            for (int i = 0; i < total_keys; ++i) {
                map_->insert(i, i * 2);
            }
        }
    }

    void TearDown(const benchmark::State& state) override {
        if (state.thread_index() == 0) {
            delete map_;
            map_ = nullptr;
        }
    }

    static ConcurrentMap* map_;
};
ConcurrentMap* ConcurrentMapFixture::map_ = nullptr;

// ---- Pure writers: every thread inserts into disjoint key ranges ----
// (disjoint ranges isolate "how fast can N threads insert in parallel"
// from CAS-contention-on-the-same-key effects, which the mixed/duplicate
// benchmark below measures separately)
BENCHMARK_DEFINE_F(ConcurrentMapFixture, WriteOnly)(benchmark::State& state) {
    // Each thread writes its own disjoint slice, offset far enough apart
    // that different threads never touch the same key.
    int keys_per_thread = static_cast<int>(state.range(0));
    int base = state.thread_index() * keys_per_thread * 1000;  // large stride, avoid overlap
    for (auto _ : state) {
        for (int i = 0; i < keys_per_thread; ++i) {
            benchmark::DoNotOptimize(map_->insert(base + i, i));
        }
    }
}
BENCHMARK_REGISTER_F(ConcurrentMapFixture, WriteOnly)
    ->Range(64, 1 << 16)
    ->ThreadRange(1, 8)
    ->UseRealTime();

// ---- Pure readers: table pre-populated in SetUp, every thread just
// looks up keys that definitely exist. Measures read-path scaling --
// should scale ~linearly with thread count since lookups take no locks
// and don't CAS.
BENCHMARK_DEFINE_F(ConcurrentMapFixture, ReadOnly)(benchmark::State& state) {
    int total_keys = static_cast<int>(state.range(0));
    std::mt19937 rng(static_cast<unsigned>(state.thread_index()) + 1);
    std::uniform_int_distribution<int> dist(0, total_keys - 1);
    for (auto _ : state) {
        int key = dist(rng);
        benchmark::DoNotOptimize(map_->at(key));
    }
}
BENCHMARK_REGISTER_F(ConcurrentMapFixture, ReadOnly)
    ->Range(64, 1 << 16)
    ->ThreadRange(1, 8)
    ->UseRealTime();

// ---- Mixed readers + writers: even thread indices read pre-populated
// keys, odd thread indices write NEW disjoint keys. Measures whether
// concurrent writers slow down concurrent readers (they shouldn't --
// lookup never CASes or blocks on a claimed slot it isn't touching).
BENCHMARK_DEFINE_F(ConcurrentMapFixture, MixedReadWrite)(benchmark::State& state) {
    int total_keys = static_cast<int>(state.range(0));
    bool is_writer = (state.thread_index() % 2) == 1;

    if (is_writer) {
        int base = 1'000'000 + state.thread_index() * 100'000;  // disjoint from pre-populated keys
        int i = 0;
        for (auto _ : state) {
            benchmark::DoNotOptimize(map_->insert(base + i, i));
            ++i;
        }
    } else {
        std::mt19937 rng(static_cast<unsigned>(state.thread_index()) + 1);
        std::uniform_int_distribution<int> dist(0, total_keys - 1);
        for (auto _ : state) {
            int key = dist(rng);
            benchmark::DoNotOptimize(map_->at(key));
        }
    }
}
BENCHMARK_REGISTER_F(ConcurrentMapFixture, MixedReadWrite)
    ->Range(64, 1 << 16)
    ->ThreadRange(2, 8)  // needs >=2 threads to have both roles present
    ->UseRealTime();

// ---- Contended writers: EVERY thread hammers the SAME small key set,
// forcing real CAS contention on the same slots. Worst case for the
// write path -- measures how badly throughput degrades under maximal
// same-slot contention rather than the disjoint-range best case above.
BENCHMARK_DEFINE_F(ConcurrentMapFixture, ContendedWrite)(benchmark::State& state) {
    constexpr int kHotKeys = 16;  // small, deliberately contended key set
    std::mt19937 rng(static_cast<unsigned>(state.thread_index()) + 1);
    std::uniform_int_distribution<int> dist(0, kHotKeys - 1);
    for (auto _ : state) {
        int key = dist(rng);
        benchmark::DoNotOptimize(map_->write(key, state.thread_index()));
    }
}
BENCHMARK_REGISTER_F(ConcurrentMapFixture, ContendedWrite)
    ->Arg(64)  // range(0) unused by this benchmark's body, but SetUp still
               // pre-populates -- kept small since only kHotKeys matter
    ->ThreadRange(1, 8)
    ->UseRealTime();

BENCHMARK_MAIN();