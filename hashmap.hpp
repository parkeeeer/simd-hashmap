/*
 * Hashtable using linear probing with simd instructions for optimizations
 * NEON simd ops are inspired by this blog:
 * https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
 * AVX ops are easier so yeah
 * supports insert delete and at functionality
 *
 * NOTE: this table doesnt support concurrent resize, a user should pretty much never use
 * concurrent=true, use ConcurrentHashMap id you want a concurrent map that supports resizing
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
#include <type_traits>
#include <utility>
#include <vector>

#include "simd_group.hpp"
#include "submap_concepts.hpp"

namespace hm {

enum class HashMapType { SingleThread, LockFree, SpinLock, Mutex, Seqlock };

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

    // Non-concurrent needs EMPTY/DELETED/OCCUPIED (2 bits, one pattern
    // to spare). Concurrent needs those plus CLAIMING (fresh insert,
    // key not yet valid to read), UPDATING (write/update in flight,
    // key stable), and MOVED (used by extract_ex/get_and_modify_meta
    // for cross-submap migration) -- one extra bit, at the cost of one
    // bit of hash tag. Computed per-instantiation so a single-threaded
    // user never pays for a state their instantiation can't reach.
    static constexpr int state_bits = concurrent ? 3 : 2;
    static constexpr int tag_bits = 8 - state_bits;
    static constexpr uint8_t tag_mask = static_cast<uint8_t>((1u << tag_bits) - 1);
    static constexpr uint8_t state_all_ones = static_cast<uint8_t>((1u << state_bits) - 1);

    static constexpr std::byte empty_mask{0b000};
    static constexpr std::byte deleted_mask{0b001};
    static constexpr std::byte claiming_mask{0b010};  // concurrent only -- fresh insert
                                                      // claiming a slot, key not yet valid
    static constexpr std::byte moved_mask{0b011};     // concurrent only -- set by
                                                      // extract_ex/get_and_modify_meta
    static constexpr std::byte updating_mask{0b100};  // concurrent only -- write/update
                                                      // in flight, key stable
    // AND-mask, not a fixed byte value like the others: occupied slots
    // carry a varying hash tag in their upper bits, so testing for
    // "occupied" means checking the low state_bits are all set,
    // regardless of what's above them.
    static constexpr std::byte occupied_mask{state_all_ones};

    static uint8_t occupied_tag(uint8_t h2) {
        return static_cast<uint8_t>(((h2 & tag_mask) << state_bits) | state_all_ones);
    }

    std::vector<tag_t> buckets;
    std::vector<std::pair<Key, Value>, allocator> values;
    Hash hasher;
    std::atomic<size_t> _size{0};
    std::atomic<size_t> load{0};  // occupied plus deleted

    // Per-slot seqlock version counter
    std::vector<std::atomic<uint32_t>> versions;

  public:
    HashMap()
        : buckets(std::vector<tag_t>(Group::width * 4)),
          values(std::vector<std::pair<Key, Value>, allocator>(Group::width * 4)),
          hasher(Hash()),
          versions(Group::width * 4) {}

    explicit HashMap(size_t min_capacity)
        : buckets(std::vector<tag_t>(((min_capacity + Group::width - 1) / Group::width) *
                                     Group::width)),
          values(std::vector<std::pair<Key, Value>, allocator>(
              ((min_capacity + Group::width - 1) / Group::width) * Group::width)),
          hasher(Hash()),
          versions(((min_capacity + Group::width - 1) / Group::width) * Group::width) {}
    // Always deleted, regardless of concurrent -- _size/load are plain
    // std::atomic<size_t> members unconditionally, and atomics are
    // neither copyable nor movable.
    HashMap(const HashMap&) = delete;
    HashMap(HashMap&&) = delete;
    ~HashMap() = default;
    HashMap& operator=(const HashMap&) = delete;
    HashMap& operator=(HashMap&&) = delete;

    bool insert(const Key& key, const Value& value) noexcept { return insert_impl(key, value); }

    bool erase(const Key& key) noexcept { return delete_impl(key); }

    bool write(const Key& key, const Value& value) noexcept {
        return write_impl(key, [&value](Value& slot) { slot = value; });
    }

    template <typename F>
    bool update(const Key& key, F&& mutator) noexcept {
        return write_impl(key, std::forward<F>(mutator));
    }

    std::optional<Value> at(const Key& key) noexcept { return at_impl(key); }

    // Public, self-contained: finds `key`, atomically erases it
    // (OCCUPIED -> DELETED), and returns what was there.
    //
    // Returns ModifyResult<Value>: check .modified, not just .found --
    // found-but-not-modified means someone else's operation (an
    // update, or a different erase) won the race on that byte. THIS
    // call didn't actually erase anything in that case, even though
    // .value would still be correct data.
    ModifyResult<Value> get_and_erase(const Key& key) noexcept {
        return get_and_modify_meta(key, deleted_mask);
    }

    // Required by SubmapLike: find, remove (OCCUPIED -> MOVED), and
    // hand back what was there -- same contract as std::map::extract.
    // The caller now owns the result and decides what to do with it,
    // including nothing. Plain optional<Value> so every submap
    // strategy can implement this identically regardless of its
    // concurrency mechanism (a mutex-based strategy can satisfy this
    // exactly as easily as this CAS-based one).
    std::optional<Value> extract(const Key& key) noexcept {
        auto result = get_and_modify_meta(key, moved_mask);
        return result.modified ? std::optional<Value>(result.value) : std::nullopt;
    }

    // Optional customization point (see HasFastExtract in
    // submap_concepts.hpp): richer than extract() above, distinguishes
    // "found and modified by us" from "found but modified by someone
    // else"
    ModifyResult<Value> extract_ex(const Key& key) noexcept {
        static_assert(concurrent, "extract_ex only makes sense under concurrent=true");
        return get_and_modify_meta(key, moved_mask);
    }

    [[nodiscard]] size_t size() const noexcept { return _size.load(); }
    [[nodiscard]] bool empty() const noexcept { return _size.load() == 0; }
    [[nodiscard]] size_t capacity() const noexcept { return buckets.size(); }
    // Raw occupied-plus-deleted count (distinct from size(), which is
    // live keys only) -- exposed so a caller like ConcurrentHashMap can
    // compute its OWN growth policy directly.
    [[nodiscard]] size_t load_count() const noexcept {
        return load.load(std::memory_order_relaxed);
    }

  private:
    void print_current_state() const {
        std::cout << "buckets: ";
        for (const auto& b : buckets) {
            std::cout << std::bitset<8>(static_cast<uint8_t>(b)) << ' ';
        }
        std::cout << '\n';
    }

    // Non-concurrent path only: plain, single-threaded rehash. No
    // races to worry about, so no CAS machinery needed at all.
    // Concurrent path never calls this (see insert_impl) -- this class
    // has no growth policy of its own under concurrent=true.
    void rehash_if_needed() {
        if (load.load(std::memory_order_relaxed) * LoadFactorDenominator >
            buckets.size() * LoadFactorNumerator) {
            size_t new_size =
                ((buckets.size() * 2 + Group::width - 1) / Group::width) * Group::width;
            std::vector<tag_t> old_buckets(std::move(buckets));
            std::vector<std::pair<Key, Value>, allocator> old_values(std::move(values));
            buckets.resize(new_size);
            values.resize(new_size);
            _size.store(0);
            load.store(0);
            for (size_t i = 0; i < old_buckets.size(); ++i) {
                if ((old_buckets[i] & occupied_mask) == occupied_mask) {
                    insert_impl(old_values[i].first, old_values[i].second);
                }
            }
        }
    }

    bool insert_impl(const Key& key, const Value& value) noexcept {
        if constexpr (!concurrent) {
            rehash_if_needed();
        }
        size_t num_groups = buckets.size() / Group::width;
        size_t hash = hasher(key);
        uint8_t h2 = static_cast<uint8_t>((hash >> 23) & tag_mask);
        size_t group_index = (hash % buckets.size()) / Group::width;
        size_t groups_probed = 0;
        [[maybe_unused]] int claim_wait_spins = 0;  // only used under concurrent=true

        while (true) {
            if constexpr (concurrent) {
                if (groups_probed >= num_groups) {
                    std::cerr << "hm::HashMap<..., concurrent=true> is "
                                 "full, every slot probed. The "
                                 "caller should have "
                                 "grown this submap well before this point. This "
                                 "should never happen; terminating.\n";
                    std::abort();
                }
            }
            size_t start_group = group_index * Group::width;
            Group group = Group::load(buckets.data() + start_group);

            for (int slot : group.match_byte(occupied_tag(h2))) {
                size_t idx = start_group + slot;
                if (values[idx].first == key) {
                    return false;  // key already exists
                }
            }

            if constexpr (concurrent) {
                // updating_mask slots: an already-occupied key is
                // mid-write/update -- its key is STABLE (mutator only
                // ever touches .second, never .first), so it's safe to
                // read and compare immediately, no waiting needed at
                // all. Resolves the duplicate question right now,
                // either way.
                for (int slot : group.match_byte(static_cast<uint8_t>(updating_mask))) {
                    size_t idx = start_group + slot;
                    if (values[idx].first == key) {
                        return false;  // duplicate -- key already exists, mid-update
                    }
                    // different key, mid-update -- nothing to wait for, just skip it
                }

                // claiming_mask slots: a fresh insert is mid-claim, key
                // not yet written -- reading it now would be a genuine
                // data race. Must wait for it to resolve (re-scan)
                // before trusting "no duplicate here," since it might
                // resolve into exactly this key. Spin (no yield) for
                // the first kSpinBeforeYield attempts -- a claim is
                // only a few instructions, so this resolves almost
                // immediately in the common case. Only fall back to
                // yield() past that, since pure unbounded spinning
                // risks starving whichever thread we're waiting on if
                // it gets preempted or scheduled onto the same core.
                if (group.match_byte(static_cast<uint8_t>(claiming_mask)).any()) {
                    constexpr int kSpinBeforeYield = 100;
                    if (++claim_wait_spins > kSpinBeforeYield) {
                        std::this_thread::yield();
                    }
                    continue;  // re-load and re-scan the SAME group
                }
            }

            simd::Bitmask empty_or_deleted{group.match_empty().raw() | group.match_deleted().raw()};
            if (empty_or_deleted.any()) {
                size_t idx = start_group + empty_or_deleted.lowest();

                if constexpr (concurrent) {
                    std::byte expected = buckets[idx].load(std::memory_order_acquire);
                    // another thread could technically come in between the simd scan and
                    // the CAS and claim this slot, so we need to recheck
                    if (expected != empty_mask && expected != deleted_mask) {
                        continue;  // no longer safe to claim -- re-scan
                    }
                    if (!buckets[idx].compare_exchange_strong(expected,
                                                              claiming_mask,
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
            ++groups_probed;
        }
    }

    std::optional<Value> at_impl(const Key& key) noexcept {
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
                        continue;  // not occupied (or claiming/updating/deleted/moved) right now
                    }
                    uint32_t v1 = versions[idx].load(std::memory_order_acquire);
                    if (v1 & 1) {
                        continue;  // a write is actively in flight -- skip, don't wait
                    }
                    bool key_matches = (values[idx].first == key);
                    Value snapshot = values[idx].second;
                    uint32_t v2 = versions[idx].load(std::memory_order_acquire);
                    if (v1 != v2) {
                        continue;  // a write happened during our read -- discard, don't trust it
                    }
                    if (key_matches) {
                        return snapshot;
                    }
                } else {
                    if (values[idx].first == key) {
                        return values[idx].second;
                    }
                }
            }
            if (group.match_empty().any()) {
                return std::nullopt;
            }
            group_index = (group_index + 1) % num_groups;
        }
        return std::nullopt;
    }

    bool delete_impl(const Key& key) noexcept {
        size_t num_groups = buckets.size() / Group::width;
        size_t hash = hasher(key);
        uint8_t h2 = static_cast<uint8_t>((hash >> 23) & tag_mask);
        size_t group_index = (hash % buckets.size()) / Group::width;
        for (size_t i = 0; i < num_groups; ++i) {
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
    bool write_impl(const Key& key, F&& mutator) noexcept {
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
                    constexpr int kSpinBeforeYield = 100;
                    int spins = 0;
                    bool claimed = false;
                    std::byte current = buckets[idx].load(std::memory_order_acquire);
                    while (true) {
                        if (current == std::byte{tag}) {
                            if (buckets[idx].compare_exchange_strong(current,
                                                                     updating_mask,
                                                                     std::memory_order_acq_rel,
                                                                     std::memory_order_relaxed)) {
                                claimed = true;
                                break;
                            }
                            // CAS failed -- `current` now holds the
                            // actual value as of that failure. Loop
                            // back around to re-evaluate it below, no
                            // extra load needed.
                        } else if (current != updating_mask) {
                            break;  // deleted or moved
                        } else {
                            // current == updating_mask: another writer
                            // transiently holds this exact slot.
                            if (++spins > kSpinBeforeYield) {
                                std::this_thread::yield();
                            }
                            current = buckets[idx].load(std::memory_order_acquire);
                        }
                    }
                    if (!claimed) {
                        break;  // key genuinely gone -- stop, not found
                    }
                    // seqlock the slot for our write, especially since mutator
                    // takes and arbitrary amount of time
                    versions[idx].fetch_add(1, std::memory_order_release);
                    mutator(values[idx].second);
                    versions[idx].fetch_add(1, std::memory_order_release);
                    buckets[idx].store(std::byte{tag}, std::memory_order_release);
                } else {
                    mutator(values[idx].second);
                }
                return true;
            }
            // NOTE: a claiming_mask or updating_mask slot here is
            // simply skipped, single key invariant handled in insert so whatever

            if (group.match_empty().any()) {
                return false;
            }
            group_index = (group_index + 1) % num_groups;
        }
        return false;
    }

    // General-purpose "find and atomically retag" primitive
    // used in get_and_erase and extract_ex. Returns a ModifyResult<Value>
    // which indicates whether the key was found, if it correctly modified,
    // and the possible value that was found. The caller can do whatever then.
    ModifyResult<Value> get_and_modify_meta(const Key& key, std::byte new_tag) noexcept {
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
                if (values[idx].first != key)
                    continue;

                std::byte expected{tag};
                if (buckets[idx].compare_exchange_strong(
                        expected, new_tag, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    _size.fetch_sub(1, std::memory_order_relaxed);
                    return ModifyResult<Value>{true, true, values[idx].second};
                }

                // Lost the race. `expected` now holds the ACTUAL
                // current byte, handed back by the failed CAS -- no
                // extra load needed to find out what it became.
                uint8_t low = static_cast<uint8_t>(expected) & state_all_ones;
                if (low == static_cast<uint8_t>(occupied_mask)) {
                    uint32_t v1 = versions[idx].load(std::memory_order_acquire);
                    if (v1 & 1) {
                        return ModifyResult<Value>{
                            false, false, Value{}};  // write happening, give up
                    }
                    Value snapshot = values[idx].second;
                    uint32_t v2 = versions[idx].load(std::memory_order_acquire);
                    if (v1 != v2) {
                        return ModifyResult<Value>{false, false, Value{}};  // torn --
                                                                            // discard
                    }
                    return ModifyResult<Value>{true, false, snapshot};
                }
                // deleted or moved (by someone else)
                return ModifyResult<Value>{false, false, Value{}};
            }
            if (group.match_empty().any()) {
                return ModifyResult<Value>{false, false, Value{}};
            }
            group_index = (group_index + 1) % num_groups;
        }
        return ModifyResult<Value>{false, false, Value{}};
    }
};

}  // namespace hm

#endif  // SIMD_HASHMAP_LIBRARY_H