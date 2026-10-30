#ifndef HM_HAZARD_POINTER_HPP
#define HM_HAZARD_POINTER_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

namespace hm {

// Cache line / coherence-unit size, used to pad hazard slots apart so
// different threads' independent publishes can never land on the
// same line and bounce between cores. Almost always 64 bytes but apple
// messes with that so we detect it at compile time. There is ptobably a better
// way to do this lol...
#if defined(__APPLE__) && defined(__aarch64__)
inline constexpr size_t kCacheLineSize = 128;
#else
inline constexpr size_t kCacheLineSize = 64;
#endif

// A minimal hazard-pointer scheme

// Flat pool of slots, generously sized, as a thread might need to protect multiple pointers
template <typename T, size_t PoolSize = 256>
class HazardPointerDomain {
  public:
    // Padded to kCacheLineSize to avoid false sharing
    struct alignas(kCacheLineSize) Slot {
        std::atomic<T*> ptr{nullptr};
        std::atomic<bool> active{false};
    };

    static HazardPointerDomain& instance() {
        static HazardPointerDomain domain;
        return domain;
    }

    Slot& acquire_slot() {
        for (auto& slot : slots_) {
            bool expected = false;
            if (slot.active.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                return slot;
            }
        }
        // Exhausted the pool this shouldn't happen, give up and abort
        std::abort();
    }

    void release_slot(Slot& slot) {
        slot.ptr.store(nullptr, std::memory_order_seq_cst);
        slot.active.store(false, std::memory_order_release);
    }

    // Is p currently hazarded by ANY active slot?
    [[nodiscard]] bool is_hazarded(T* p) const {
        for (const auto& slot : slots_) {
            if (slot.active.load(std::memory_order_acquire) &&
                slot.ptr.load(std::memory_order_seq_cst) == p) {
                return true;
            }
        }
        return false;
    }

  private:
    std::array<Slot, PoolSize> slots_;
};

// RAII guard for the reader side. Construct one per pointer you need
// to hold simultaneously (not one per thread).
template <typename T>
class HazardPointer {
  public:
    HazardPointer() : slot_(HazardPointerDomain<T>::instance().acquire_slot()) {}
    ~HazardPointer() { HazardPointerDomain<T>::instance().release_slot(slot_); }

    HazardPointer(const HazardPointer&) = delete;
    HazardPointer& operator=(const HazardPointer&) = delete;

    // Publish-and-confirm: reads `source`, publishes it into this
    // slot, then re-reads `source` to confirm it hasn't changed
    // underneath us. If it has, the just-published value can't be
    // trusted -- retry with whatever it changed to.
    //
    // Uses seq_cst (not just acquire/release) on both the publish and
    // the re-read: acquire/release alone doesn't prevent a
    // store-then-load on one thread from being observed out of order
    // by a mirroring load-then-store on another (the classic
    // Dekker's-algorithm hazard) -- a full fence on both the
    // publish here AND the writer's scan (see RetireList::reclaim
    // below) is what actually closes that gap. This is the one place
    // where being conservative matters, and it's cheap: it happens
    // once per submap touched during a chain walk, not once per byte
    // scanned the way the hot inner loops are.
    T* protect(const std::atomic<T*>& source) {
        T* p = source.load(std::memory_order_seq_cst);
        while (true) {
            slot_.ptr.store(p, std::memory_order_seq_cst);
            T* p2 = source.load(std::memory_order_seq_cst);
            if (p2 == p) {
                return p;  // confirmed stable -- safe to dereference
            }
            p = p2;  // source changed underneath us -- retry with the new value
        }
    }

    void reset() { slot_.ptr.store(nullptr, std::memory_order_seq_cst); }

    // Publishes an ALREADY-known-valid pointer into this slot,
    // without reading or re-confirming against any atomic source.
    // Safe specifically because the caller already holds a hazard on
    // `p` via some OTHER slot at the moment this is called (e.g.
    // transferring "curr" to become the new "prev" while walking a
    // chain) -- there's nothing to race against since nothing can
    // free `p` while it's still hazarded somewhere.
    void adopt(T* p) { slot_.ptr.store(p, std::memory_order_seq_cst); }

  private:
    typename HazardPointerDomain<T>::Slot& slot_;
};

// Writer-side deferred reclamation: objects that have been unlinked
// but might still be hazarded by some in-flight reader. retire()
// stashes the pointer and opportunistically frees anything in the
// list that's no longer hazarded by anyone.
template <typename T>
class RetireList {
  public:
    void retire(T* p) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.push_back(p);
        reclaim();
    }

    ~RetireList() {
        // Best-effort final cleanup. Anything still hazarded at
        // destruction time is a caller bug (using the structure past
        // its own lifetime), not something to paper over -- just
        // leak it rather than risk a use-after-free on shutdown.
        std::lock_guard<std::mutex> lock(mutex_);
        auto& domain = HazardPointerDomain<T>::instance();
        for (T* p : pending_) {
            if (!domain.is_hazarded(p))
                delete p;
        }
    }

  private:
    void reclaim() {
        auto& domain = HazardPointerDomain<T>::instance();
        pending_.erase(std::remove_if(pending_.begin(),
                                      pending_.end(),
                                      [&](T* p) {
                                          if (!domain.is_hazarded(p)) {
                                              delete p;
                                              return true;
                                          }
                                          return false;
                                      }),
                       pending_.end());
    }

    std::mutex mutex_;
    std::vector<T*> pending_;
};

}  // namespace hm

#endif  // HM_HAZARD_POINTER_HPP