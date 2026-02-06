#include <gtest/gtest.h>
#include "hashmap.h"

TEST(HashMapTest, BasicInsertAndFind) {
    hashmap::HashMap<int, std::string> map;

    EXPECT_TRUE(map.insert(1, "one"));
    EXPECT_EQ(map.at(1), "one");
    EXPECT_EQ(map.size(), 1);
}

TEST(HashMapTest, InsertDuplicate) {
    hashmap::HashMap<int, int> map;

    EXPECT_TRUE(map.insert(5, 10));
    EXPECT_FALSE(map.insert(5, 20));
    EXPECT_EQ(map.at(5), 10);
}

TEST(HashMapTest, Erase) {
    hashmap::HashMap<int, int> map;

    map.insert(1, 100);
    EXPECT_TRUE(map.erase(1));
    EXPECT_THROW(map.at(1), std::out_of_range);
}

TEST(HashMapTest, MultipleInserts) {
    hashmap::HashMap<int, int> map;

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
    hashmap::HashMap<int, int> map;

    map.insert(1, 100);
    map.erase(1);
    EXPECT_TRUE(map.insert(1, 200));
    EXPECT_EQ(map.at(1), 200);
}