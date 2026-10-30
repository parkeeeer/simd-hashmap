#include <benchmark/benchmark.h>

#include <algorithm>
#include <atomic>
#include <random>
#include <unordered_map>

#include "ConcurrentHashMap.hpp"
#include "hashmap.hpp"

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

using ConcurrentMap = hm::HashMap<int, int, /*concurrent=*/true>;

class ConcurrentMapFixture : public benchmark::Fixture {
  public:
    void SetUp(const benchmark::State& state) override {
        if (state.thread_index() == 0) {
            map_ = new ConcurrentMap();
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

BENCHMARK_DEFINE_F(ConcurrentMapFixture, WriteOnly)(benchmark::State& state) {
    int keys_per_thread = static_cast<int>(state.range(0));
    int base = state.thread_index() * keys_per_thread * 1000;
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

BENCHMARK_DEFINE_F(ConcurrentMapFixture, MixedReadWrite)(benchmark::State& state) {
    int total_keys = static_cast<int>(state.range(0));
    bool is_writer = (state.thread_index() % 2) == 1;
    if (is_writer) {
        int base = 1'000'000 + state.thread_index() * 100'000;
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
    ->ThreadRange(2, 8)
    ->UseRealTime();

BENCHMARK_DEFINE_F(ConcurrentMapFixture, ContendedWrite)(benchmark::State& state) {
    constexpr int kHotKeys = 16;
    std::mt19937 rng(static_cast<unsigned>(state.thread_index()) + 1);
    std::uniform_int_distribution<int> dist(0, kHotKeys - 1);
    for (auto _ : state) {
        int key = dist(rng);
        benchmark::DoNotOptimize(map_->write(key, state.thread_index()));
    }
}
BENCHMARK_REGISTER_F(ConcurrentMapFixture, ContendedWrite)
    ->Arg(64)
    ->ThreadRange(1, 8)
    ->UseRealTime();

// ===========================================================================
// hm::ConcurrentHashMap benchmarks
//
// Same Fixture idiom as ConcurrentMapFixture above, focuses more on migration
// Stuff
// ===========================================================================

using ConcurrentHashMap = hm::ConcurrentHashMap<int, int>;

class ConcurrentHashMapFixture : public benchmark::Fixture {
  public:
    void SetUp(const benchmark::State& state) override {
        if (state.thread_index() == 0) {
            chm_ = new ConcurrentHashMap();
            int total_keys = static_cast<int>(state.range(0));
            for (int i = 0; i < total_keys; ++i) {
                chm_->insert(i, i * 2);
            }
        }
    }
    void TearDown(const benchmark::State& state) override {
        if (state.thread_index() == 0) {
            delete chm_;
            chm_ = nullptr;
        }
    }
    static ConcurrentHashMap* chm_;
};
ConcurrentHashMap* ConcurrentHashMapFixture::chm_ = nullptr;

BENCHMARK_DEFINE_F(ConcurrentHashMapFixture, WriteOnly)(benchmark::State& state) {
    int keys_per_thread = static_cast<int>(state.range(0));
    int base = state.thread_index() * keys_per_thread * 1000;
    for (auto _ : state) {
        for (int i = 0; i < keys_per_thread; ++i) {
            benchmark::DoNotOptimize(chm_->insert(base + i, i));
        }
    }
}
BENCHMARK_REGISTER_F(ConcurrentHashMapFixture, WriteOnly)
    ->Range(64, 1 << 14)
    ->ThreadRange(1, 8)
    ->UseRealTime();

// ---- Pure reads on pre-populated, already-settled keys (all at head,
// SetUp finished before any thread starts) -- the "nothing to migrate"
// baseline, comparable to ConcurrentMapFixture's ReadOnly.
BENCHMARK_DEFINE_F(ConcurrentHashMapFixture, ReadOnlySettled)(benchmark::State& state) {
    int total_keys = static_cast<int>(state.range(0));
    std::mt19937 rng(static_cast<unsigned>(state.thread_index()) + 1);
    std::uniform_int_distribution<int> dist(0, total_keys - 1);
    for (auto _ : state) {
        int key = dist(rng);
        benchmark::DoNotOptimize(chm_->get(key));
    }
}
BENCHMARK_REGISTER_F(ConcurrentHashMapFixture, ReadOnlySettled)
    ->Range(64, 1 << 14)
    ->ThreadRange(1, 8)
    ->UseRealTime();

// ---- Reads deliberately picked from the OLDEST slice of inserted keys
// only -- these live deepest in the chain, isolating migration overhead
// specifically rather than mixing it with already-settled reads.
BENCHMARK_DEFINE_F(ConcurrentHashMapFixture, ReadWithMigration)(benchmark::State& state) {
    int total_keys = static_cast<int>(state.range(0));
    int oldest_slice = std::max(1, total_keys / 10);  // oldest 10% of inserted keys
    std::mt19937 rng(static_cast<unsigned>(state.thread_index()) + 1);
    std::uniform_int_distribution<int> dist(0, oldest_slice - 1);
    for (auto _ : state) {
        int key = dist(rng);
        benchmark::DoNotOptimize(chm_->get(key));
    }
}
BENCHMARK_REGISTER_F(ConcurrentHashMapFixture, ReadWithMigration)
    ->Range(1024, 1 << 14)  // needs enough keys to have grown past one submap
    ->ThreadRange(1, 8)
    ->UseRealTime();

// ---- Contended writers/updates, same small hot-key-set shape as
// ConcurrentMapFixture's ContendedWrite, at the ConcurrentHashMap level.
BENCHMARK_DEFINE_F(ConcurrentHashMapFixture, ContendedWrite)(benchmark::State& state) {
    constexpr int kHotKeys = 16;
    std::mt19937 rng(static_cast<unsigned>(state.thread_index()) + 1);
    std::uniform_int_distribution<int> dist(0, kHotKeys - 1);
    for (auto _ : state) {
        int key = dist(rng);
        benchmark::DoNotOptimize(chm_->write(key, state.thread_index()));
    }
}
BENCHMARK_REGISTER_F(ConcurrentHashMapFixture, ContendedWrite)
    ->Arg(64)
    ->ThreadRange(1, 8)
    ->UseRealTime();

BENCHMARK_MAIN();