/*
 * Hashtable using linear probing with simd instructions for optimizations
 * NEON simd ops are inspired by this blog:
 * https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
 * AVX ops are easier so yeah
 * supports insert delete and at functionality
 */

#ifndef SIMD_HASHMAP_LIBRARY_H
#define SIMD_HASHMAP_LIBRARY_H
#include <atomic>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "simd_group.hpp"

namespace hm {

template <typename Key,
          typename Value,
          bool concurrent = true,
          size_t LoadFactorNumerator = 7,
          size_t LoadFactorDenominator = 8,
          typename Hash = std::hash<Key>,
          typename allocator = std::allocator<std::pair<Key, Value>>>
class HashMap {
    static_assert(!concurrent || std::is_trivially_copyable_v<Value>,
                  "hm::HashMap requires trivially copyable Value types under "
                  "concurrent access -- large/owning types risk torn reads. "
                  "Store a pointer, index, or shared_ptr<T> instead.");
 
    using tag_t = std::conditional_t<concurrent, std::atomic<std::byte>, std::byte>;
    using Group = simd::Group;
 
    // Non-concurrent only ever needs EMPTY/DELETED/OCCUPIED (2 bits,
    // one pattern to spare). Concurrent needs those plus IN_PROGRESS
    // and (eventually) MOVED -- one extra bit, at the cost of one bit
    // of hash tag. Computed per-instantiation so a single-threaded user
    // never pays for a state their instantiation can't reach.
    static constexpr int state_bits = concurrent ? 3 : 2;
    static constexpr int tag_bits = 8 - state_bits;
    static constexpr uint8_t tag_mask = static_cast<uint8_t>((1u << tag_bits) - 1);
    static constexpr uint8_t state_all_ones = static_cast<uint8_t>((1u << state_bits) - 1);
 
    static constexpr std::byte empty_mask{0b000};
    static constexpr std::byte deleted_mask{0b001};
    static constexpr std::byte in_progress_mask{0b010};  // concurrent only
    static constexpr std::byte moved_mask{0b011};        // concurrent only, not yet wired up
 
    static uint8_t occupied_tag(uint8_t h2) {
        return static_cast<uint8_t>(((h2 & tag_mask) << state_bits) | state_all_ones);
    }
 
    std::vector<tag_t> buckets;
    std::vector<std::pair<Key, Value>, allocator> values;
    Hash hasher;
    std::atomic<size_t> _size{0};
    std::atomic<size_t> load{0};  // occupied plus deleted
 
  public:
    HashMap()
        : buckets(std::vector<tag_t>(Group::width * 4)),
          values(std::vector<std::pair<Key, Value>, allocator>(Group::width * 4)),
          hasher(Hash()) {}
    HashMap(const HashMap&) = default;
    HashMap(HashMap&&) noexcept = default;
    ~HashMap() = default;
    HashMap& operator=(const HashMap&) = default;
    HashMap& operator=(HashMap&&) noexcept = default;
 
    bool insert(const Key& key, const Value& value) { return insert_impl(key, value); }
 
    bool erase(const Key& key) noexcept { return delete_impl(key); }
 
    bool write(const Key& key, const Value& value) {
        return write_impl(key, [&value](Value& slot) { slot = value; });
    }
 
    template <typename F>
    bool update(const Key& key, F&& mutator) {
        return write_impl(key, std::forward<F>(mutator));
    }
 
    std::optional<Value> at(const Key& key) noexcept { return at_impl(key); }
 
    [[nodiscard]] size_t size() const noexcept { return _size.load(); }
    [[nodiscard]] bool empty() const noexcept { return _size.load() == 0; }
 
  private:
    void print_current_state() const {
        std::cout << "buckets: ";
        for (const auto& b : buckets) {
            std::cout << std::bitset<8>(static_cast<uint8_t>(b)) << ' ';
        }
        std::cout << '\n';
    }
 
    // Concurrent path: resizing was moved up to ConcurrentHashMap's
    // chain-of-submaps design entirely. A submap here is meant to be
    // fixed-capacity -- if this fires, it means either the map wasn't
    // sized appropriately up front, or something is using this class
    // directly instead of through the chain wrapper.
    //
    // Non-concurrent path: plain, single-threaded rehash. No races to
    // worry about, so no CAS/chunk machinery needed at all.
    void rehash_if_needed() {
        if (load.load(std::memory_order_relaxed) * LoadFactorDenominator >
            buckets.size() * LoadFactorNumerator) {
            if constexpr (concurrent) {
                throw std::logic_error(
                    "hm::HashMap<..., concurrent=true> attempted to resize -- "
                    "concurrent resize is not supported. Size the map "
                    "appropriately up front, or use hm::ConcurrentHashMap "
                    "for unbounded growth.");
            } else {
                size_t new_size =
                    ((buckets.size() * 2 + Group::width - 1) / Group::width) * Group::width;
                std::vector<tag_t> old_buckets(std::move(buckets));
                std::vector<std::pair<Key, Value>, allocator> old_values(std::move(values));
                buckets.resize(new_size);
                values.resize(new_size);
                _size.store(0);
                load.store(0);
                for (size_t i = 0; i < old_buckets.size(); ++i) {
                    uint8_t low_bits = static_cast<uint8_t>(old_buckets[i]) & state_all_ones;
                    if (low_bits == state_all_ones) {  // occupied
                        insert_impl(old_values[i].first, old_values[i].second);
                    }
                }
            }
        }
    }
 
    bool insert_impl(const Key& key, const Value& value) {
        rehash_if_needed();
        size_t num_groups = buckets.size() / Group::width;
        size_t hash = hasher(key);
        uint8_t h2 = static_cast<uint8_t>((hash >> 23) & tag_mask);
        size_t group_index = (hash % buckets.size()) / Group::width;
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
 
                if constexpr (concurrent) {
                    std::byte expected = buckets[idx].load(std::memory_order_acquire);
                    if (!buckets[idx].compare_exchange_strong(expected,
                                                              in_progress_mask,
                                                              std::memory_order_acq_rel,
                                                              std::memory_order_relaxed)) {
                        continue;
                    }
                    // can safely own idx now
                    values[idx] = std::make_pair(key, value);
                    buckets[idx].store(std::byte{occupied_tag(h2)}, std::memory_order_release);
                } else {
                    buckets[idx] = std::byte{occupied_tag(h2)};
                    values[idx] = std::make_pair(key, value);
                }
                _size.fetch_add(1, std::memory_order_relaxed);
                load.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            group_index = (group_index + 1) % num_groups;
        }
    }
 
    std::optional<Value> at_impl(const Key& key) {
        // NOTE: lookups do NOT call rehash_if_needed -- reads should
        // never throw or trigger a resize. If this were ever wired to
        // the concurrent path, a lookup finding the load factor tripped
        // just means "somebody needs to grow this soon," not something
        // a read should act on itself.
        size_t num_groups = buckets.size() / Group::width;
        size_t hash = hasher(key);
        uint8_t h2 = static_cast<uint8_t>((hash >> 23) & tag_mask);
        uint8_t tag = occupied_tag(h2);
        size_t group_index = (hash % buckets.size()) / Group::width;
        for (size_t i = 0; i < num_groups; ++i) {
            size_t start_group = group_index * Group::width;
            Group group = Group::load(buckets.data() + start_group);
 
            for (int slot : group.match_byte(tag)) {
                size_t idx = start_group + slot;
                if constexpr (concurrent) {
                    if (buckets[idx].load(std::memory_order_acquire) != std::byte{tag}) {
                        continue;
                    }
                }
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
        uint8_t h2 = static_cast<uint8_t>((hash >> 23) & tag_mask);
        size_t group_index = (hash % buckets.size()) / Group::width;
        for (size_t i = 0; i < num_groups; ++i) {
            size_t start_group = group_index * Group::width;
            Group group = Group::load(buckets.data() + start_group);
 
            for (int slot : group.match_byte(occupied_tag(h2))) {
                size_t idx = start_group + slot;
                if (values[idx].first != key) continue;
                if constexpr (concurrent) {
                    std::byte expected{occupied_tag(h2)};
                    if (!buckets[idx].compare_exchange_strong(expected,
                                                              deleted_mask,
                                                              std::memory_order_acq_rel,
                                                              std::memory_order_relaxed)) {
                        break;
                    }
                } else {
                    buckets[idx] = deleted_mask;
                }
                _size.fetch_sub(1, std::memory_order_relaxed);
                return true;
            }
            if (group.match_empty().any()) {
                return false;
            }
            group_index = (group_index + 1) % num_groups;
        }
        return false;
    }
 
    template <typename F>
    bool write_impl(const Key& key, F&& mutator) {
        size_t num_groups = buckets.size() / Group::width;
        size_t hash = hasher(key);
        uint8_t h2 = static_cast<uint8_t>((hash >> 23) & tag_mask);
        uint8_t tag = occupied_tag(h2);
        size_t group_index = (hash % buckets.size()) / Group::width;
 
        for (size_t i = 0; i < num_groups; ++i) {
            size_t start = group_index * Group::width;
            Group group = Group::load(buckets.data() + start);
 
            for (int slot : group.match_byte(tag)) {
                size_t idx = start + slot;
                if (values[idx].first != key) {
                    continue;
                }
 
                if constexpr (concurrent) {
                    std::byte current = buckets[idx].load(std::memory_order_acquire);
                    if (current != std::byte{tag}) {
                        break;
                    }
                    if (!buckets[idx].compare_exchange_strong(current,
                                                              in_progress_mask,
                                                              std::memory_order_acq_rel,
                                                              std::memory_order_relaxed)) {
                        break;
                    }
                    mutator(values[idx].second);
                    buckets[idx].store(std::byte{tag}, std::memory_order_release);
                } else {
                    mutator(values[idx].second);
                }
                return true;
            }
 
            if (group.match_empty().any()) {
                return false;
            }
            group_index = (group_index + 1) % num_groups;
        }
        return false;
    }
};

struct no_migrate_t {};
inline constexpr no_migrate_t no_migrate{};

template<typename Key, 
typename Value, 
size_t LoadFactorNum = 7, 
size_t LoadFactorDenom = 8,
typename Hash = std::hash<Key>, 
typename allocator = std::allocator<std::pair<Key, Value>>> 
class ConcurrentHashMap{
    struct submap {
        std::atomic<std::shared_ptr<submap>> next;
        HashMap<Key,Value, true, 10, 1, Hash, allocator> map;
    };
    std::atomic<std::shared_ptr<submap>> head;
    std::atomic<size_t> num_submaps;

    public:

    ConcurrentHashMap();
    ConcurrentHashMap(size_t init_size);

    void set_background_thread(size_t ms = 0);//will do bg work, mostly migrating things
    bool insert(const Key& key, const Value& value) noexcept;
    bool erase(const Key& key) noexcept;
    bool write(const Key& key, const Value& value) noexcept;
    bool write(const Key& key, const Value& value, no_migrate_t) noexcept;
    bool update(const Key& key, const Value& value) noexcept;
    bool update(const Key& key, const Value& value, no_migrate_t) noexcept;
    std::optional<Value> get(const Key& key) noexcept;
    std::optional<Value> get(const Key& key, no_migrate_t) noexcept;

    size_t size();

};
}  // namespace hm

#endif  // SIMD_HASHMAP_LIBRARY_H