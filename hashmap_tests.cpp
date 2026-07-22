#include <gtest/gtest.h>

#include "hashmap.hpp"

TEST(HashMapTest, BasicInsertAndFind) {
    hm::HashMap<int, std::string> map;

    EXPECT_TRUE(map.insert(1, "one"));
    EXPECT_EQ(map.at(1), "one");
    EXPECT_EQ(map.size(), 1);
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
    hm::HashMap<int, int> map;

    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(map.insert(i, i * 10));
    }

    for (int i = 0; i < 100; ++i) {
        try {
            EXPECT_EQ(map.at(i), i * 10);
        } catch (std::out_of_range&) {
            std::cout << "Failed to find key: " << i << "\n";
            std::cout << "Map size: " << map.size() << "\n";
            throw;
        }
    }
}

TEST(HashMapTest, EraseAndReinsert) {
    hm::HashMap<int, int> map;

    map.insert(1, 100);
    map.erase(1);
    EXPECT_TRUE(map.insert(1, 200));
    EXPECT_EQ(map.at(1), 200);
}