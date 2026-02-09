/*
 * Hashtable using linear probing with simd instructions for optimizations
 * NEON simd ops are inspired by this blog: https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
 * AVX ops are easier and just written by me
 * supports insert delete and at functionality
 */







#ifndef SIMD_HASHMAP_LIBRARY_H
#define SIMD_HASHMAP_LIBRARY_H
#include <functional>
#include <iostream>
#include <vector>
#include <bitset>
#include <cstdint>
#include <utility>



#ifdef __ARM_NEON
#define HASH_HAS_NEON 1
#include <arm_neon.h>
using match_type = uint64_t;
#define HASH_SIMD_SIZE 16
#elif defined(__AVX2__)  //avx2 is needed for the 8 bit comparison ops
#define HASH_HAS_AVX2 1
#include <immintrin.h>
using match_type = uint32_t
#define HASH_SIMD_SIZE 32
#else
#error "needs simd support"
#endif

namespace hashmap {

    template<typename Key, typename Value,
    typename Hash = std::hash<Key>,
    typename allocator = std::allocator<std::pair<Key, Value>>>
    class HashMap {

        // all and masks
        static constexpr std::byte deleted_mask{0b01};
        static constexpr std::byte empty_mask{0b10};
        static constexpr std::byte occupied_mask{0b11};
        std::vector<std::byte> buckets;
        std::vector<std::pair<Key, Value>, allocator> values;
        Hash hasher;
        size_t _size{0};
        size_t load{0}; //occupied plus deleted
    public:
        HashMap() : buckets(std::vector<std::byte>(64, empty_mask)), hasher(Hash()), values(std::vector<std::pair<Key, Value>, allocator>(64)) {}
        HashMap(const HashMap&) = default;
        HashMap(HashMap&&) noexcept = default;
        ~HashMap() = default;
        HashMap& operator=(const HashMap&) = default;
        HashMap& operator=(HashMap&&) noexcept = default;

        bool insert(const Key& key, const Value& value){return insert_impl(key, value);}

        bool erase(const Key& key) {return delete_impl(key);}

        Value& at(const Key& key){return at_impl(key);}
        const Value& at(const Key& key) const {return at_impl(key);}

        [[nodiscard]] size_t size() const {return _size;}
        [[nodiscard]] bool empty() const {return _size == 0;}


    private:
        void print_current_state() const {
            std::cout << "buckets: ";
            for (const auto& b : buckets) {
                std::cout << std::bitset<8>(static_cast<uint8_t>(b)) << ' ';
            }
            std::cout << '\n';
        }

        void rehash_if_needed() {
            if (load * 8 > buckets.capacity() * 7) {
                size_t new_size = ((buckets.capacity() * 2 + HASH_SIMD_SIZE - 1) / HASH_SIMD_SIZE) * HASH_SIMD_SIZE;
                std::vector<std::byte> old_buckets(std::move(buckets));
                std::vector<std::pair<Key, Value>, allocator> old_values(std::move(values));
                buckets.resize(new_size, empty_mask);
                values.resize(new_size);
                _size = 0;
                load = 0;
                for (size_t i{0};i < old_buckets.size();++i) {
                    if ((old_buckets[i] & occupied_mask) == occupied_mask) {
                        insert_impl(old_values[i].first, old_values[i].second);
                    }
                }
            }
        }

#if HASH_HAS_NEON
        match_type vec_to_bitmask(uint8x16_t vec) {
            const uint16x8_t equalMask = vreinterpretq_u16_u8(vec);
            const uint8x8_t res = vshrn_n_u16(equalMask, 4);
            return vget_lane_u64(vreinterpret_u64_u8(res), 0);
        }
#elif HASH_HAS_AVX2
#endif

        match_type match_hash(const std::byte* group, uint8_t h2) {
#if HASH_HAS_NEON
            uint8x16_t vec = vld1q_u8(reinterpret_cast<const uint8_t*>(group));
            uint8_t target = (h2 << 2) | 3;
            uint8x16_t target_vec = vdupq_n_u8(target);
            uint8x16_t cmp = vceqq_u8(vec, target_vec);
            return vec_to_bitmask(cmp);
#endif
        }

        match_type match_empty(const std::byte* group) {
#if HASH_HAS_NEON
            uint8x16_t vec = vld1q_u8(reinterpret_cast<const uint8_t*>(group));
            uint8_t target = 0b10;
            uint8x16_t target_vec = vdupq_n_u8(target);
            //vec = vandq_u8(vec, target_vec);
            uint8x16_t cmp = vceqq_u8(vec, target_vec);
            return vec_to_bitmask(cmp);
#endif
        }

        match_type match_deleted(const std::byte* group) {
#if HASH_HAS_NEON
            uint8x16_t vec = vld1q_u8(reinterpret_cast<const uint8_t*>(group));
            uint8_t target = 0b01;
            uint8x16_t target_vec = vdupq_n_u8(target);
            //vec = vandq_u8(vec, target_vec);
            uint8x16_t cmp = vceqq_u8(vec, target_vec);
            return vec_to_bitmask(cmp);
#endif
        }


        bool insert_impl(const Key& key, const Value& value) {
            rehash_if_needed();
            size_t num_groups = buckets.capacity() / HASH_SIMD_SIZE;
            size_t hash = hasher(key);
            size_t index = hash % buckets.size();
            uint8_t h2 = (hash >> 23) & 0b111111;
            uint8_t insertion = h2 << 2 | 3;
            size_t group_index = index / HASH_SIMD_SIZE;
            while (true) {
                size_t start_group = group_index * HASH_SIMD_SIZE;
                const std::byte* group = buckets.data() + start_group;

                match_type mask = match_hash(group, h2);
                while (mask) {
                    int slot = __builtin_ctzll(mask) >> 2;
                    size_t idx = start_group + slot;
                    if (values[idx].first == key) {
                        return false; //key exists
                    }
                    mask &= mask-1;
                }

                match_type empty_match = match_empty(group);
                match_type deleted_match = match_deleted(group);
                mask = empty_match | deleted_match;
                if (mask) {
                    int slot = __builtin_ctzll(mask) >> 2;
                    size_t idx = start_group + slot;
                    buckets[idx] = std::byte{insertion};
                    values[idx] = std::make_pair(key, value);
                    ++_size;
                    ++load;
                    return true;
                }
                group_index = (group_index + 1) % num_groups;
            }
        }

        Value& at_impl(const Key& key) {
            size_t num_groups = buckets.capacity() / HASH_SIMD_SIZE;
            size_t hash = hasher(key);
            size_t index = hash % buckets.size();
            uint8_t h2 = (hash >> 23) & 0b111111;
            size_t group_index = index / HASH_SIMD_SIZE;
            for (size_t i{0};i < num_groups;++i) {
                size_t start_group = group_index * HASH_SIMD_SIZE;
                const std::byte* group = buckets.data() + start_group;
                match_type mask = match_hash(group, h2);
                while (mask) {
                    int slot = __builtin_ctzll(mask) >> 2;
                    size_t idx = start_group + slot;
                    if (values[idx].first == key) {
                        return values[idx].second;
                    }
                    mask &= mask-1;
                }
                match_type empty = match_empty(group);
                if (empty) {
                    throw std::out_of_range("Key not found");
                }
                group_index = (group_index + 1) % num_groups;
            }
            throw std::out_of_range("key not found");
        }

        bool delete_impl(const Key& key) {
            size_t num_groups = buckets.capacity() / HASH_SIMD_SIZE;
            size_t hash = hasher(key);
            size_t index = hash % buckets.size();
            uint8_t h2 = (hash >> 23) & 0b111111;
            size_t group_index = index / HASH_SIMD_SIZE;
            for (size_t i{0}; i < num_groups; ++i) {
                size_t start_group = group_index * HASH_SIMD_SIZE;
                const std::byte* group = buckets.data() + start_group;
                match_type mask = match_hash(group, h2);
                if (!mask) {
                    return false;
                }
                while (mask) {
                    int slot = __builtin_ctzll(mask) >> 2;
                    size_t idx = start_group + slot;
                    if (values[idx].first == key) {
                        buckets[idx] = deleted_mask;
                        --_size;
                        return true;
                    }
                    mask &= mask-1;
                }

                match_type empty = match_empty(group);
                if (empty) {
                    return false;
                }
                group_index = (group_index + 1) % num_groups;
            }
            return false;
        }
    };
}

#endif // SIMD_HASHMAP_LIBRARY_H