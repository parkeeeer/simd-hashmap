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
          size_t GroupsPerChunk = 8,
          typename Hash = std::hash<Key>,
          typename allocator = std::allocator<std::pair<Key, Value>>>
class HashMap {
    static_assert(!concurrent || std::is_trivially_copyable_v<Value>,
                  "hm::HashMap requires trivially copyable Value types under "
                  "concurrent access -- large/owning types risk torn reads. "
                  "Store a pointer, index, or shared_ptr<T> instead.");
    static_assert((GroupsPerChunk & (GroupsPerChunk - 1)) == 0,
                  "GroupsPerChunk should be a power of two so ChunkSize divides "
                  "any power-of-two capacity produced by doubling");
    using tag_t = std::conditional_t<concurrent, std::atomic<std::byte>, std::byte>;
    using Group = simd::Group;
    static constexpr std::byte deleted_mask{0b01};
    static constexpr std::byte empty_mask{0b00};
    static constexpr std::byte occupied_mask{0b11};
    static constexpr std::byte in_progress_mask{0b10};
    std::vector<tag_t> buckets;
    std::vector<std::pair<Key, Value>, allocator> values;
    Hash hasher;
    std::atomic<size_t> _size{0};
    std::atomic<size_t> load{0};  // occupied plus deleted
  public:
    HashMap()
        : buckets(std::vector<tag_t>(Group::width * GroupsPerChunk)),
          hasher(Hash()),
          values(std::vector<std::pair<Key, Value>, allocator>(Group::width * GroupsPerChunk)) {}
    HashMap(const HashMap&) = default;
    HashMap(HashMap&&) noexcept = default;
    ~HashMap() = default;
    HashMap& operator=(const HashMap&) = default;
    HashMap& operator=(HashMap&&) noexcept = default;

    bool insert(const Key& key, const Value& value) noexcept { return insert_impl(key, value); }

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

    void rehash_if_needed() {
        if (load.load() * 8 > buckets.size() * 7) {
            if constexpr (concurrent) {
                try_init_resize();
            } else {
                size_t new_size =
                    ((buckets.size() * 2 + Group::width - 1) / Group::width) * Group::width;
                std::vector<tag_t> old_buckets(std::move(buckets));
                std::vector<std::pair<Key, Value>, allocator> old_values(std::move(values));
                buckets.resize(new_size);
                values.resize(new_size);
                _size.store(0);
                load.store(0);
                for (size_t i{0}; i < old_buckets.size(); ++i) {
                    if ((old_buckets[i] & occupied_mask) == occupied_mask) {
                        insert_impl(old_values[i].first, old_values[i].second);
                    }
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
        rehash_if_needed();
        size_t num_groups = buckets.size() / Group::width;
        size_t hash = hasher(key);
        size_t index = hash % buckets.size();
        uint8_t h2 = (hash >> 23) & 0b111111;
        uint8_t tag = occupied_tag(h2);
        size_t group_index = index / Group::width;
        for (size_t i{0}; i < num_groups; ++i) {
            size_t start_group = group_index * Group::width;
            Group group = Group::load(buckets.data() + start_group);

            for (int slot : group.match_byte(occupied_tag(h2))) {
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
        rehash_if_needed();
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
                if (values[idx].first != key)
                    continue;
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
        rehash_if_needed();
        size_t num_groups = buckets.size() / Group::width;
        size_t hash = hasher(key);
        uint8_t h2 = static_cast<uint8_t>((hash >> 23) & 0b111111);
        uint8_t tag = occupied_tag(h2);
        size_t group_index = (hash % buckets.size()) / Group::width;

        for (size_t i{0}; i < num_groups; ++i) {
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

    //----EVERYTHING BELOW THIS LINE IS JUST RESIZE LOGIC(NOT FUN)----//

    struct ResizeState {
        std::vector<std::atomic<std::byte>> new_buckets;
        std::vector<std::pair<Key, Value>> new_values;
        std::atomic<size_t> next_chunk;
        std::atomic<size_t> chunks_remaining;
    };

    std::atomic<std::shared_ptr<ResizeState>> active_resize{nullptr};

    // this function will set up the ResizeState so that the hash table can be correctly rehashed
    // and avoid race conditions
    void try_init_resize() {
        std::shared_ptr<ResizeState> state = active_resize.load(std::memory_order_acquire);
        if (!state) {
            // no resize started, try to set it up
            size_t num_chunks = buckets.size() / (Group::width * GroupsPerChunk);
            std::shared_ptr<ResizeState> candidate = std::make_shared<ResizeState>(
                std::vector<std::atomic<std::byte>>(buckets.size() * 2),
                std::vector<std::pair<Key, Value>>(buckets.size() * 2),
                num_chunks,
                num_chunks);

            std::shared_ptr<ResizeState> expected = nullptr;

            if (active_resize.compare_exchange_strong(
                    expected, candidate, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                // we won!!!!
                state = candidate;
            } else {
                state = expected;
            }
        }

        help_migrate(state);
    }

    void help_migrate(std::shared_ptr<ResizeState> state) {
        while (true) {
            size_t claimed = state->next_chunk.fetch_sub(1, std::memory_order_acq_rel);
            if (claimed == 0) {
                state->next_chunk.fetch_add(1, std::memory_order_relaxed);
                while (active_resize.load(std::memory_order_acquire) == state) {
                    std::this_thread::yield();
                }
                return;
            }

            size_t chunk_id = claimed - 1;
            migrate_all_slots_in_chunk(*state, chunk_id);
            if (state->chunks_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                finish_resize(*state);
                return;
            }
        }
    }

    void migrate_all_slots_in_chunk(ResizeState& state, size_t chunk_id) {
        size_t start = chunk_id * GroupsPerChunk * Group::width;

        for (size_t g{0}; g < GroupsPerChunk; ++g) {
            size_t group_start = start + g * Group::width;
            Group group = Group::load(buckets.data() + group_start);

            for (int slot : group.match_occupied()) {
                size_t old_idx = group_start + slot;

                constexpr int max_attempts = 64;
                for (int attempt = 0; attempt < max_attempts; ++attempt) {
                    std::byte current = buckets[old_idx].load(std::memory_order_acquire);
                    std::byte low2 = (current)&std::byte(0b11);

                    if (low2 != occupied_mask) {
                        if (current == in_progress_mask) {
                            std::this_thread::yield();
                            continue;
                        }
                        break;
                    }

                    if (buckets[old_idx].compare_exchange_strong(current,
                                                                 in_progress_mask,
                                                                 std::memory_order_acq_rel,
                                                                 std::memory_order_relaxed)) {
                        const auto& [key, value] = values[old_idx];
                        insert_into_new_table(state, key, value);
                        break;
                    }

                    if (attempt == max_attempts - 1) {
                        assert(false && "migration retry exhausted, lost data");
                    }
                }
            }
        }
    }

    void insert_into_new_table(ResizeState& state, const Key& key, const Value& value) {
        size_t num_groups = state.new_buckets.size() / Group::width;
        size_t hash = hasher(key);
        uint8_t h2 = static_cast<uint8_t>((hash >> 23) & 0b111111);
        size_t group_index = (hash % state.new_buckets.size()) / Group::width;

        while (true) {
            size_t start = group_index * Group::width;
            Group group = Group::load(state.new_buckets.data() + start);

            simd::Bitmask empty = group.match_empty();
            if (empty.any()) {
                size_t idx = start + empty.lowest();
                std::byte expected{empty_mask};
                if (state.new_buckets[idx].compare_exchange_strong(expected,
                                                                   in_progress_mask,
                                                                   std::memory_order_acq_rel,
                                                                   std::memory_order_relaxed)) {
                    state.new_values[idx] = std::make_pair(key, value);
                    state.new_buckets[idx].store(std::byte{occupied_tag(h2)},
                                                 std::memory_order_release);
                    return;
                }
                continue;
            }
            group_index = (group_index + 1) % num_groups;
        }
    }

    void finish_resize(ResizeState& state) {
        buckets = std::move(state.new_buckets);
        values = std::move(state.new_values);

        active_resize.store(nullptr, std::memory_order_release);
    }
};
}  // namespace hm

#endif  // SIMD_HASHMAP_LIBRARY_H