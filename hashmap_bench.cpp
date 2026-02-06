#include <benchmark/benchmark.h>
#include <unordered_map>
#include "hashmap.h"
#include <random>


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
BENCHMARK(BM_StdHashMap_Insert)->Range(64, 1<<20)->Complexity();


// Benchmark insert
static void BM_YourHashMap_Insert(benchmark::State& state) {
    std::mt19937 rng(42);
    for (auto _ : state) {
        hashmap::HashMap<int, int> map;
        for (int i = 0; i < state.range(0); ++i) {
            map.insert(rng(), i);
        }
        benchmark::DoNotOptimize(map);
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_YourHashMap_Insert)->Range(64, 1<<20)->Complexity();

// Benchmark lookup
static void BM_YourHashMap_Lookup(benchmark::State& state) {
    hashmap::HashMap<int, int> map;
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
BENCHMARK(BM_YourHashMap_Lookup)->Range(64, 1<<20)->Complexity();

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
BENCHMARK(BM_StdHashMap_Lookup)->Range(64, 1<<20)->Complexity();

BENCHMARK_MAIN();