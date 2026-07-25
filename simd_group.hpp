#ifndef HASHMAP_SIMD_GROUP_HPP
#define HASHMAP_SIMD_GROUP_HPP

/**
 * Simd abstraction file for the hashmap
 * I don't want the hashmap file to have any
 * intrinsics in it so I made this file to abstract it away
 */

#include <cstddef>
#include <cstdint>

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#else
#error "need simd support(NEON or AVX2)"
#endif

namespace hm::simd {

#ifdef __ARM_NEON
constexpr bool _hasneon = true;
constexpr bool _hasavx2 = false;
#elif defined(__AVX2__)
constexpr bool _hasneon = false;
constexpr bool _hasavx2 = true;
#endif
class Bitmask {
  public:
#ifdef __ARM_NEON
    using raw_type = uint64_t;
    static constexpr int BitsPerSlot = 4;
#elif defined(__AVX2__)
    using raw_type = uint32_t;
    static constexpr int BitsPerSlot = 1;
#endif

    explicit Bitmask(raw_type mask) : mask(mask) {}

    [[nodiscard]] bool any() const { return mask != 0; }
    [[nodiscard]] raw_type raw() const { return mask; }

    [[nodiscard]] int lowest() const {
        if constexpr (BitsPerSlot == 1) {
            return __builtin_ctz(mask);
        } else {
            return __builtin_ctzll(mask) / BitsPerSlot;
        }
    }

    void clear_lowest() {
        if constexpr (BitsPerSlot == 1) {
            mask &= mask - 1;
        } else {
            raw_type lsb_slot_mask = static_cast<raw_type>(0xF) << (lowest() * BitsPerSlot);
            mask &= ~lsb_slot_mask;
        }
    }

  private:
    raw_type mask;
};

class BitmaskIterator {
  public:
    explicit BitmaskIterator(Bitmask m) : mask(m) {}
    int operator*() const { return mask.lowest(); }
    BitmaskIterator& operator++() {
        mask.clear_lowest();
        return *this;
    }
    bool operator==(const BitmaskIterator& other) const { return mask.raw() == other.mask.raw(); }
    bool operator!=(const BitmaskIterator& other) const { return !(*this == other); }

  private:
    Bitmask mask;
};

inline BitmaskIterator begin(Bitmask m) { return BitmaskIterator(m); }
inline BitmaskIterator end(Bitmask) { return BitmaskIterator(Bitmask(0)); }

class Group {
  public:
#ifdef __ARM_NEON
    using vec_type = uint8x16_t;
    static constexpr size_t width = 16;
#elif defined(__AVX2__)
    using vec_type = __m256i;
    static constexpr size_t width = 32;
#endif

    static Group load(const std::byte* pos) {
#ifdef __ARM_NEON
        return Group{vld1q_u8(reinterpret_cast<const uint8_t*>(pos))};
#elif defined(__AVX2__)
        return Group{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pos))};
#endif
    }

    [[nodiscard]] Bitmask match_byte(uint8_t tag) const {
#ifdef __ARM_NEON
        uint8x16_t target = vdupq_n_u8(tag);
        uint8x16_t cmp = vceqq_u8(vec, target);
        return Bitmask{to_nibble_mask(cmp)};
#elif defined(__AVX2__)
        __m256i target = _mm256_set1_epi8(static_cast<char>(tag));
        __m256i cmp = _mm256_cmpeq_epi8(vec, target);
        return Bitmask{static_cast<uint32_t>(_mm256_movemask_epi8(cmp))};
#endif
    }

    [[nodiscard]] Bitmask match_empty() const { return match_byte(0b10); }
    [[nodiscard]] Bitmask match_deleted() const { return match_byte(0b01); }

  private:
    explicit Group(vec_type v) : vec(v) {}
    vec_type vec;

#ifdef __ARM_NEON
    static uint64_t to_nibble_mask(uint8x16_t cmp) {
        const uint16x8_t widened = vreinterpretq_u16_u8(cmp);
        const uint8x8_t narrowed = vshrn_n_u16(widened, 4);
        return vget_lane_u64(vreinterpret_u64_u8(narrowed), 0);
    }
#endif
};
}  // namespace hm::simd

#endif