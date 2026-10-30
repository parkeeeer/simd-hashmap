#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "ConcurrentHashMap.hpp"
#include "hashmap.hpp"

// ===========================================================================
// hm::HashMap -- single-threaded usage of both instantiations.
// (hm::HashMap<int,int> defaults concurrent=true, so these already cover
// the concurrent code path under single-threaded use; the first test
// explicitly uses concurrent=false to cover that path too.)
// ===========================================================================

TEST(HashMapTest, BasicInsertAndFind) {
    hm::HashMap<int, std::string, false> map;

    EXPECT_TRUE(map.insert(1, "one"));
    EXPECT_EQ(map.at(1), "one");
    EXPECT_EQ(map.size(), 1u);  // 1u: size() returns size_t, avoids a sign-compare warning
}

TEST(HashMapTest, InsertDuplicate) {
    hm::HashMap<int, int> map;

    EXPECT_TRUE(map.insert(5, 10));
    EXPECT_FALSE(map.insert(5, 20));
    EXPECT_EQ(map.at(5), 10);
}

TEST(HashMapTest, Erase) {
    hm::HashMap<int, int> map;

    map.insert(1, 100);
    EXPECT_TRUE(map.erase(1));
    EXPECT_FALSE(map.at(1));
}

TEST(HashMapTest, MultipleInserts) {
    hm::HashMap<int, int> map(1024);

    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(map.insert(i, i * 10));
    }

    // NOTE: at() returns std::optional<Value> and never throws --
    // there's no std::out_of_range to catch anymore. EXPECT_EQ against
    // an optional<Value> directly (comparing to the raw value) already
    // fails with a clear diagnostic on a miss, so no try/catch needed.
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(map.at(i), i * 10) << "missing key: " << i << ", map size: " << map.size();
    }
}

TEST(HashMapTest, EraseAndReinsert) {
    hm::HashMap<int, int> map;

    map.insert(1, 100);
    map.erase(1);
    EXPECT_TRUE(map.insert(1, 200));
    EXPECT_EQ(map.at(1), 200);
}

TEST(HashMapTest, BasicWrite) {
    hm::HashMap<int, int> map;
    map.insert(1, 100);
    EXPECT_EQ(map.at(1), 100);
    map.write(1, 120);
    EXPECT_EQ(map.at(1), 120);
}

TEST(HashMapTest, BasicUpdate) {
    hm::HashMap<int, int> map;
    map.insert(1, 100);
    EXPECT_TRUE(map.update(1, [](int& v) { v += 20; }));
    EXPECT_EQ(map.at(1), 120);
    EXPECT_FALSE(map.update(999, [](int& v) { v += 1; }));  // key doesn't exist
}

TEST(HashMapTest, ConcurrentResizeAborts) {
    // concurrent=true has no growth policy of its own -- genuine
    // exhaustion is a hard std::abort(), not a throw. Confirms it
    // fires (rather than silently corrupting or looping forever) and
    // that it does NOT fire for the non-concurrent path, which resizes
    // normally instead.
    using SmallConcurrentMap = hm::HashMap<int, int, true>;  // type alias avoids
                                                             // template commas
                                                             // confusing the
                                                             // EXPECT_DEATH macro
    EXPECT_DEATH(
        {
            SmallConcurrentMap map(32);
            for (int i = 0; i < 1000; ++i)
                map.insert(i, i);
        },
        "");

    hm::HashMap<int, int, false> map(32);
    for (int i = 0; i < 1000; ++i) {
        map.insert(i, i);  // should just resize normally, no abort
    }
    EXPECT_EQ(map.size(), 1000u);
}

// ===========================================================================
// hm::HashMap -- real multi-threaded stress tests.
// (Previously entirely absent despite this being a concurrent structure.)
// ===========================================================================

TEST(HashMapConcurrentTest, DisjointConcurrentInserts) {
    hm::HashMap<int, int, true> map(8192);  // pre-sized, avoids any abort risk
    constexpr int kThreads = 8;
    constexpr int kPerThread = 300;
    std::atomic<int> successes{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&map, &successes, t]() {
            for (int i = 0; i < kPerThread; ++i) {
                int key = t * kPerThread + i;
                if (map.insert(key, key * 2))
                    successes.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads)
        th.join();
    EXPECT_EQ(successes.load(), kThreads * kPerThread);

    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kPerThread; ++i) {
            int key = t * kPerThread + i;
            auto v = map.at(key);
            ASSERT_TRUE(v.has_value()) << "missing key: " << key;
            EXPECT_EQ(*v, key * 2);
        }
    }
}

TEST(HashMapConcurrentTest, ManyThreadsSameKeyInsert_ExactlyOneWins) {
    hm::HashMap<int, int, true> map(64);
    constexpr int kThreads = 16;
    std::atomic<int> successes{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&map, &successes, t]() {
            if (map.insert(42, t))
                successes.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& th : threads)
        th.join();

    EXPECT_EQ(successes.load(), 1);
    EXPECT_EQ(map.size(), 1u);
}

TEST(HashMapConcurrentTest, ConcurrentUpdatesPreserveSum) {
    hm::HashMap<int, int, true> map(64);
    map.insert(1, 0);
    constexpr int kThreads = 8;
    constexpr int kPerThread = 100;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&map]() {
            for (int i = 0; i < kPerThread; ++i) {
                bool ok = map.update(1, [](int& v) { v += 1; });
                for (int retry = 0; !ok && retry < 100; ++retry) {
                    ok = map.update(1, [](int& v) { v += 1; });
                }
            }
        });
    }
    for (auto& th : threads)
        th.join();

    EXPECT_EQ(map.at(1), kThreads * kPerThread);
}

// ===========================================================================
// hm::ConcurrentHashMap -- basic single-threaded correctness.
// (Previously zero coverage despite the header being included.)
// ===========================================================================

TEST(ConcurrentHashMapTest, BasicInsertGetErase) {
    hm::ConcurrentHashMap<int, int> chm;

    EXPECT_TRUE(chm.insert(1, 100));
    EXPECT_FALSE(chm.insert(1, 999));  // duplicate rejected
    EXPECT_EQ(chm.get(1), 100);
    EXPECT_TRUE(chm.erase(1));
    EXPECT_FALSE(chm.get(1).has_value());
    EXPECT_FALSE(chm.erase(1));  // already gone
}

TEST(ConcurrentHashMapTest, WriteAndUpdate) {
    hm::ConcurrentHashMap<int, int> chm;
    chm.insert(1, 100);

    EXPECT_TRUE(chm.write(1, 555));
    EXPECT_EQ(chm.get(1), 555);

    EXPECT_TRUE(chm.update(1, [](int& v) { v += 1000; }));
    EXPECT_EQ(chm.get(1), 1555);

    EXPECT_FALSE(chm.write(999, 1));  // write/update don't upsert
}

TEST(ConcurrentHashMapTest, NoMigrateVariants) {
    hm::ConcurrentHashMap<int, int> chm;
    chm.insert(1, 100);

    EXPECT_EQ(chm.get(1, hm::no_migrate), 100);
    EXPECT_TRUE(chm.write(1, 200, hm::no_migrate));
    EXPECT_EQ(chm.get(1, hm::no_migrate), 200);
    EXPECT_TRUE(chm.update(1, [](int& v) { v += 1; }, hm::no_migrate));
    EXPECT_EQ(chm.get(1, hm::no_migrate), 201);
}

TEST(ConcurrentHashMapTest, GrowthAcrossMultipleSubmaps) {
    hm::ConcurrentHashMap<int, int> chm;
    // Default submap starts small -- this forces several real
    // prepend_submap growth events.
    for (int i = 0; i < 800; ++i) {
        ASSERT_TRUE(chm.insert(i, i * 2)) << "failed at i=" << i;
    }
    EXPECT_EQ(chm.size(), 800u);
    for (int i = 0; i < 800; ++i) {
        auto v = chm.get(i);
        ASSERT_TRUE(v.has_value()) << "missing key: " << i;
        EXPECT_EQ(*v, i * 2);
    }
}

TEST(ConcurrentHashMapTest, MigrationViaGetKeepsDataCorrect) {
    hm::ConcurrentHashMap<int, int> chm;
    for (int i = 0; i < 800; ++i)
        chm.insert(i, i);
    // Early keys are now sitting in older, non-head submaps. Access
    // them repeatedly via get() (which migrates on a non-head hit) and
    // confirm they stay correct and eventually reachable in one probe.
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < 50; ++i) {
            auto v = chm.get(i);
            ASSERT_TRUE(v.has_value());
            EXPECT_EQ(*v, i);
        }
    }
    EXPECT_EQ(chm.size(), 800u);
}

TEST(ConcurrentHashMapTest, WriteUpdateOnMigratedKeys) {
    hm::ConcurrentHashMap<int, int> chm;
    for (int i = 0; i < 800; ++i)
        chm.insert(i, i);
    // key 3 is now deep in the chain, not at head
    EXPECT_TRUE(chm.write(3, 30000));
    EXPECT_EQ(chm.get(3), 30000);
    EXPECT_TRUE(chm.update(4, [](int& v) { v += 100000; }));
    EXPECT_EQ(chm.get(4), 4 + 100000);
    EXPECT_EQ(chm.size(), 800u);
}

// ===========================================================================
// hm::ConcurrentHashMap -- multi-threaded stress tests.
// ===========================================================================

TEST(ConcurrentHashMapConcurrentTest, ConcurrentInsertsAndReads) {
    hm::ConcurrentHashMap<int, int> chm;
    constexpr int kThreads = 8;
    constexpr int kPerThread = 300;
    std::atomic<int> successes{0};

    std::vector<std::thread> writers;
    for (int t = 0; t < kThreads; ++t) {
        writers.emplace_back([&chm, &successes, t]() {
            for (int i = 0; i < kPerThread; ++i) {
                int key = t * kPerThread + i;
                if (chm.insert(key, key * 2))
                    successes.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : writers)
        th.join();
    ASSERT_EQ(successes.load(), kThreads * kPerThread);

    std::atomic<int> found{0};
    std::atomic<int> mismatched{0};
    std::vector<std::thread> readers;
    for (int t = 0; t < kThreads; ++t) {
        readers.emplace_back([&chm, &found, &mismatched, t]() {
            for (int i = 0; i < kPerThread; ++i) {
                int key = t * kPerThread + i;
                auto v = chm.get(key, 500);  // generous max_attempts under heavy contention
                if (v.has_value()) {
                    found.fetch_add(1, std::memory_order_relaxed);
                    if (*v != key * 2)
                        mismatched.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& th : readers)
        th.join();

    EXPECT_EQ(found.load(), kThreads * kPerThread);
    EXPECT_EQ(mismatched.load(), 0);
}

TEST(ConcurrentHashMapConcurrentTest, ConcurrentUpdatesRacingMigration) {
    hm::ConcurrentHashMap<int, int> chm;
    for (int i = 0; i < 500; ++i)
        chm.insert(i, 0);

    constexpr int kThreads = 6;
    constexpr int kPerThread = 200;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&chm]() {
            for (int rep = 0; rep < kPerThread; ++rep) {
                int key = rep % 500;
                bool ok = chm.update(key, [](int& v) { v += 1; });
                for (int retry = 0; !ok && retry < 100; ++retry) {
                    ok = chm.update(key, [](int& v) { v += 1; });
                }
                chm.get(key % 100);  // force migration traffic alongside the updates
            }
        });
    }
    for (auto& th : threads)
        th.join();

    long total = 0;
    for (int i = 0; i < 500; ++i) {
        auto v = chm.get(i);
        ASSERT_TRUE(v.has_value());
        total += *v;
    }
    EXPECT_EQ(total, kThreads * kPerThread);
}

TEST(ConcurrentHashMapConcurrentTest, ManyThreadsSameKeyInsert_ExactlyOneWins) {
    hm::ConcurrentHashMap<int, int> chm;
    constexpr int kThreads = 16;
    std::atomic<int> successes{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&chm, &successes, t]() {
            if (chm.insert(7, t))
                successes.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& th : threads)
        th.join();

    EXPECT_EQ(successes.load(), 1);
    EXPECT_EQ(chm.size(), 1u);
}