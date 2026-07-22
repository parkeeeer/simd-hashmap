/*
 * Hashtable using linear probing with simd instructions for optimizations
 * NEON simd ops are inspired by this blog:
 * https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
 * AVX ops are easier so yeah
 * supports insert delete and at functionality
 */

#ifndef SIMD_HASHMAP_LIBRARY_H
#define SIMD_HASHMAP_LIBRARY_H
#include <bitset>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <utility>
#include <vector>

#include "simd_group.hpp"

namespace hm {

template <typename Key,
          typename Value,
          typename Hash = std::hash<Key>,
          typename allocator = std::allocator<std::pair<Key, Value>>>
class HashMap {
    // all and masks

    using Group = simd::Group;
    static constexpr std::byte deleted_mask{0b01};
    static constexpr std::byte empty_mask{0b10};
    static constexpr std::byte occupied_mask{0b11};
    static constexpr std::byte in_progress_mask{0b00};
    std::vector<std::byte> buckets;
    std::vector<std::pair<Key, Value>, allocator> values;
    Hash hasher;
    size_t _size{0};
    size_t load{0};  // occupied plus deleted
  public:
    HashMap()
        : buckets(std::vector<std::byte>(Group::width * 4, empty_mask)),
          hasher(Hash()),
          values(std::vector<std::pair<Key, Value>, allocator>(Group::width * 4)) {}
    HashMap(const HashMap&) = default;
    HashMap(HashMap&&) noexcept = default;
    ~HashMap() = default;
    HashMap& operator=(const HashMap&) = default;
    HashMap& operator=(HashMap&&) noexcept = default;

    bool insert(const Key& key, const Value& value) noexcept { return insert_impl(key, value); }

    bool erase(const Key& key) noexcept { return delete_impl(key); }

    std::optional<Value> at(const Key& key) noexcept { return at_impl(key); }

    [[nodiscard]] size_t size() const noexcept { return _size; }
    [[nodiscard]] bool empty() const noexcept { return _size == 0; }

  private:
    void print_current_state() const {
        std::cout << "buckets: ";
        for (const auto& b : buckets) {
            std::cout << std::bitset<8>(static_cast<uint8_t>(b)) << ' ';
        }
        std::cout << '\n';
    }

    void rehash_if_needed() {
        if (load * 8 > buckets.size() * 7) {
            size_t new_size =
                ((buckets.size() * 2 + Group::width - 1) / Group::width) * Group::width;
            std::vector<std::byte> old_buckets(std::move(buckets));
            std::vector<std::pair<Key, Value>, allocator> old_values(std::move(values));
            buckets.resize(new_size, empty_mask);
            values.resize(new_size);
            _size = 0;
            load = 0;
            for (size_t i{0}; i < old_buckets.size(); ++i) {
                if ((old_buckets[i] & occupied_mask) == occupied_mask) {
                    insert_impl(old_values[i].first, old_values[i].second);
                }
            }
        }
    }

    static uint8_t occupied_tag(uint8_t h2) { return static_cast<uint8_t>((h2 << 2) | 0b11); }

    bool insert_impl(const Key& key, const Value& value) {
        rehash_if_needed();
        size_t num_groups = buckets.size() / Group::width;
        size_t hash = hasher(key);
        size_t index = hash % buckets.size();
        uint8_t h2 = (hash >> 23) & 0b111111;
        uint8_t insertion = h2 << 2 | 3;
        size_t group_index = index / Group::width;
        while (true) {
            size_t start_group = group_index * Group::width;
            Group group = Group::load(buckets.data() + start_group);

            for (int slot : group.match_byte(occupied_tag(h2))) {
                size_t idx = start_group + slot;
                if (values[idx].first == key) {
                    return false;  // key already exists
                }
            }

            simd::Bitmask empty_or_deleted{group.match_empty().raw() | group.match_deleted().raw()};
            if (empty_or_deleted.any()) {
                size_t idx = start_group + empty_or_deleted.lowest();
                buckets[idx] = std::byte{occupied_tag(h2)};
                values[idx] = std::make_pair(key, value);
                ++_size;
                ++load;
                return true;
            }
            group_index = (group_index + 1) % num_groups;
        }
    }

    std::optional<Value> at_impl(const Key& key) {
        size_t num_groups = buckets.size() / Group::width;
        size_t hash = hasher(key);
        size_t index = hash % buckets.size();
        uint8_t h2 = (hash >> 23) & 0b111111;
        size_t group_index = index / Group::width;
        for (size_t i{0}; i < num_groups; ++i) {
            size_t start_group = group_index * Group::width;
            Group group = Group::load(buckets.data() + start_group);

            for (int slot : group.match_byte(occupied_tag(h2))) {
                size_t idx = start_group + slot;
                if (values[idx].first == key) {
                    return values[idx].second;
                }
            }
            if (group.match_empty().any()) {
                return std::nullopt;
            }
            group_index = (group_index + 1) % num_groups;
        }
        return std::nullopt;
    }

    bool delete_impl(const Key& key) {
        size_t num_groups = buckets.size() / Group::width;
        size_t hash = hasher(key);
        size_t index = hash % buckets.size();
        uint8_t h2 = (hash >> 23) & 0b111111;
        size_t group_index = index / Group::width;
        for (size_t i{0}; i < num_groups; ++i) {
            size_t start_group = group_index * Group::width;
            Group group = Group::load(buckets.data() + start_group);

            for (int slot : group.match_byte(occupied_tag(h2))) {
                size_t idx = start_group + slot;
                if (values[idx].first == key) {
                    buckets[idx] = deleted_mask;
                    --_size;
                    return true;
                }
            }
            if (group.match_empty().any()) {
                return false;
            }
            group_index = (group_index + 1) % num_groups;
        }
        return false;
    }
};
}  // namespace hm

#endif  // SIMD_HASHMAP_LIBRARY_H