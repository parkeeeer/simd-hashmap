#ifndef HM_CONCURRENT_HASHMAP_HPP
#define HM_CONCURRENT_HASHMAP_HPP

// history of design attempts lol:
//
//  v1: std::atomic<std::shared_ptr<submap>>. Turns out this is not even
//  lock free on libstd++, and throughput was really bad on any more than one thread
//  attempting access at once.
//
//  v2: plain std::shared_ptr guarded by std::shared_mutex. Made it a little better
//  but still scaled really badly, especially a huge jump from 1 to 2 threads.
//  I thought that this would make reading cheaper(since most operations are reads),
//  but I guess not.
//
//  v3 (this version): did some research and found hazard pointers, which is an alternate way
//  to manage lifetimes rather than ref counts, and it works way better, much faster

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>

#include "hazard_pointer.hpp"

// hashmap.hpp isnt really needed, just for include convenience...
#include "hashmap.hpp"
#include "submap_concepts.hpp"

namespace hm {

struct no_migrate_t {};
inline constexpr no_migrate_t no_migrate{};
struct no_check_t {};
inline constexpr no_check_t no_check{};

template <
    typename Key,
    typename Value,
    typename SubmapType =
        HashMap<Key, Value, true, 10, 1, std::hash<Key>, std::allocator<std::pair<Key, Value>>>,
    size_t LoadFactorNum = 7,
    size_t LoadFactorDenom = 8,
    bool DefaultNoMigrate = false>
    requires SubmapLike<SubmapType, Key, Value>
class ConcurrentHashMap {
    struct submap {
        std::atomic<submap*> next{nullptr};
        SubmapType map;

        submap() : map() {}
        explicit submap(size_t capacity) : map(capacity) {}
    };

    std::atomic<submap*> head;
    RetireList<submap> retire_list_;

    // Detected once per instantiation, used quite often with `if constexpr`
    static constexpr bool kHasFastExtract = HasFastExtract<SubmapType, Key, Value>;

    // hazard-pointer-protected accessors, replacing the old
    // shared_ptr/shared_mutex call sites

    [[nodiscard]] submap* protect_head(HazardPointer<submap>& hp) const { return hp.protect(head); }

    [[nodiscard]] submap* protect_next(HazardPointer<submap>& hp, submap* s) const {
        return hp.protect(s->next);
    }

  public:
    ConcurrentHashMap() : head(new submap()) {}
    explicit ConcurrentHashMap(size_t init_size) : head(new submap(init_size)) {}

    ~ConcurrentHashMap() {
        // Single-threaded teardown context (the object is being
        // destroyed, so no concurrent access is possible) -- safe to
        // walk and delete directly without any hazard-pointer
        // protection here.
        submap* curr = head.load(std::memory_order_relaxed);
        while (curr) {
            submap* next = curr->next.load(std::memory_order_relaxed);
            delete curr;
            curr = next;
        }
    }
    ConcurrentHashMap(const ConcurrentHashMap&) = delete;
    ConcurrentHashMap& operator=(const ConcurrentHashMap&) = delete;

    void set_background_thread(size_t ms = 0);  // will do bg work, mostly migrating things

    bool insert(const Key& key, const Value& value, size_t max_attempts = 2) noexcept {
        // Search the entire chain to avoid duplicate keys, then insert into head.
        HazardPointer<submap> hp;
        auto curr = protect_head(hp);
        while (curr) {
            if (curr->map.at(key))
                return false;  // already exists somewhere
            curr = protect_next(hp, curr);
        }
        return insert(key, value, no_check, max_attempts);
    }

    bool insert(const Key& key, const Value& value, no_check_t, size_t max_attempts = 2) noexcept {
        HazardPointer<submap> hp;
        for (size_t i = 0; i < max_attempts; ++i) {
            auto h = protect_head(hp);
            // A conforming submap has no growth policy of its own
            // (fixed-capacity, see SubmapLike's docs) -- this check
            // MUST happen before every insert attempt, since insert()
            // returning false only ever means "duplicate key," never
            // "out of room."
            if (h->map.load_count() * LoadFactorDenom > h->map.capacity() * LoadFactorNum) {
                prepend_submap();  // don't care if we won or lost the race --
                                   // either way head has grown by the time
                                   // we loop back and re-check
                continue;
            }
            if (h->map.insert(key, value))
                return true;
        }
        return false;
    }

    bool erase(const Key& key) noexcept {
        HazardPointer<submap> hp;
        auto curr = protect_head(hp);
        while (curr) {
            if (curr->map.erase(key))
                return true;
            curr = protect_next(hp, curr);
        }
        return false;
    }

    bool write(const Key& key, const Value& value, size_t max_attempts = 2) noexcept {
        return write_impl(key, max_attempts, [&value](Value& v) { v = value; });
    }

    // no_migrate variant: only ever touches wherever the key currently
    // lives, never promotes it toward head. Purely a latency knob for
    // callers who can't tolerate the occasional migration CAS.
    bool write(const Key& key, const Value& value, no_migrate_t) noexcept {
        HazardPointer<submap> hp;
        auto curr = protect_head(hp);
        while (curr) {
            if (curr->map.update(key, [&value](Value& v) { v = value; }))
                return true;
            curr = protect_next(hp, curr);
        }
        return false;
    }

    template <typename F>
    bool update(const Key& key, F&& mutator, size_t max_attempts = 2) noexcept {
        return write_impl(key, max_attempts, std::forward<F>(mutator));
    }

    template <typename F>
    bool update(const Key& key, F&& mutator, no_migrate_t) noexcept {
        HazardPointer<submap> hp;
        auto curr = protect_head(hp);
        while (curr) {
            if (curr->map.update(key, mutator))
                return true;
            curr = protect_next(hp, curr);
        }
        return false;
    }

    std::optional<Value> get(const Key& key, size_t max_attempts = 2) noexcept {
        if constexpr (DefaultNoMigrate) {
            return get(key, hm::no_migrate);
        }
        HazardPointer<submap> hp;
        for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
            auto curr = protect_head(hp);
            if (!curr)
                return std::nullopt;

            if (auto v = curr->map.at(key))
                return v;  // already at front, done

            curr = protect_next(hp, curr);
            while (curr) {
                if constexpr (kHasFastExtract) {
                    auto result = curr->map.extract_ex(key);
                    if (result.found) {
                        if (result.modified) {
                            insert_into_head(key, result.value);
                        }
                        return result.value;
                    }
                } else {
                    // Generic fallback: plain extract() already means
                    // "found and removed by this call" or nullopt --
                    // no ambiguous middle state to resolve.
                    if (auto v = curr->map.extract(key)) {
                        insert_into_head(key, *v);
                        return v;
                    }
                }

                if (curr->map.size() == 0) {
                    try_remove_submap(curr);
                }
                curr = protect_next(hp, curr);
            }
            // Not found anywhere this pass
        }
        return std::nullopt;
    }

    // Pure read: never migrates, never mutates anything.
    std::optional<Value> get(const Key& key, no_migrate_t) noexcept {
        HazardPointer<submap> hp;
        auto curr = protect_head(hp);
        while (curr) {
            if (auto v = curr->map.at(key))
                return v;
            curr = protect_next(hp, curr);
        }
        return std::nullopt;
    }

    // Live sum across the chain, not a maintained counter -- same
    // reasoning as choosing not to maintain a submap-count field.
    [[nodiscard]] size_t size() const noexcept {
        size_t total = 0;
        HazardPointer<submap> hp;
        auto curr = protect_head(hp);
        while (curr) {
            total += curr->map.size();
            curr = protect_next(hp, curr);
        }
        return total;
    }

  private:
    // Shared shape for write()/update(): if the key is already at
    // head, apply the mutator there directly, no migration needed. If
    // it's found further back, extract it, apply the mutator locally,
    // and insert the mutated result into head.
    template <typename F>
    bool write_impl(const Key& key, size_t max_attempts, F&& mutator) noexcept {
        HazardPointer<submap> hp;
        for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
            auto curr = protect_head(hp);
            if (!curr)
                return false;

            if (curr->map.update(key, mutator)) {
                return true;  // already at head, no migration involved at all
            }

            curr = protect_next(hp, curr);
            while (curr) {
                if constexpr (kHasFastExtract) {
                    auto result = curr->map.extract_ex(key);
                    if (result.found) {
                        if (result.modified) {
                            Value new_value = result.value;
                            mutator(new_value);
                            insert_into_head(key, new_value);
                            return true;
                        }
                        break;  // found but modified by someone else -- retry fresh
                    }
                } else {
                    if (auto v = curr->map.extract(key)) {
                        Value new_value = *v;
                        mutator(new_value);
                        insert_into_head(key, new_value);
                        return true;
                    }
                }
                if (curr->map.size() == 0) {
                    try_remove_submap(curr);
                }
                curr = protect_next(hp, curr);
            }
        }
        return false;
    }

    void insert_into_head(const Key& key, const Value& value) noexcept {
        HazardPointer<submap> hp;
        while (true) {
            auto h = protect_head(hp);
            if (h->map.load_count() * LoadFactorDenom > h->map.capacity() * LoadFactorNum) {
                prepend_submap();
                continue;
            }
            h->map.insert(key, value);
            return;
        }
    }

    // walk from head, done right now, specifically for this
    // removal. if the CAS loses a race with another
    // concurrent removal, we just don't remove anything this pass.
    // Actual deletion is deferred to retire_list_, target might
    // still be hazarded even after
    // being unlinked here.
    void try_remove_submap(submap* target) {
        HazardPointer<submap> hp_prev;
        HazardPointer<submap> hp_curr;

        auto h = protect_head(hp_prev);
        if (h == target)
            return;  // never remove head

        submap* prev = h;
        submap* curr = protect_next(hp_curr, prev);
        while (curr) {
            if (curr == target) {
                submap* next_of_curr = curr->next.load(std::memory_order_acquire);
                submap* expected = curr;
                if (prev->next.compare_exchange_strong(expected,
                                                       next_of_curr,
                                                       std::memory_order_acq_rel,
                                                       std::memory_order_relaxed)) {
                    retire_list_.retire(curr);
                }
                return;
            }

            prev = curr;
            hp_prev.adopt(prev);
            curr = protect_next(hp_curr, curr);
        }
    }

    // Appends a new, bigger submap ahead of the current head.
    // Returns true if this call won the race and published it, false if someone else grew head
    // first (caller should just re-check head and proceed).
    bool prepend_submap() {
        HazardPointer<submap> hp;
        auto h = protect_head(hp);
        auto candidate = new submap(h->map.capacity() * 2);
        candidate->next.store(h, std::memory_order_relaxed);

        submap* expected = h;
        if (head.compare_exchange_strong(
                expected, candidate, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return true;
        }
        // lost the race
        delete candidate;

        return false;
    }
};

}  // namespace hm

#endif