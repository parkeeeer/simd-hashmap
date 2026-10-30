#ifndef HM_SUBMAP_CONCEPTS_HPP
#define HM_SUBMAP_CONCEPTS_HPP

#include <concepts>
#include <cstddef>
#include <optional>

namespace hm {

// result to indicate modifications on a key value pair
template <typename Value>
struct ModifyResult {
    bool modified = false;
    bool found = false;
    Value value{};
};

// The required contract every submap plugin must satisfy. Deliberately
// minimal
template <typename M, typename Key, typename Value>
concept SubmapLike = std::constructible_from<M, size_t> && std::default_initializable<M> &&
                     requires(M m, const Key& k, const Value& v) {
                         { m.insert(k, v) } -> std::same_as<bool>;
                         { m.at(k) } -> std::same_as<std::optional<Value>>;
                         { m.erase(k) } -> std::same_as<bool>;
                         { m.extract(k) } -> std::same_as<std::optional<Value>>;
                         {
                             m.update(k, [](Value&) {})
                         } -> std::same_as<bool>;
                         { m.size() } -> std::same_as<size_t>;
                         { m.capacity() } -> std::same_as<size_t>;
                         { m.load_count() } -> std::same_as<size_t>;
                     };

// Optional customization point: a submap MAY additionally provide
// extract_ex(), a richer version of extract() that resolves "found and
// modified by us" vs "found but modified by someone else" vs
// "not found" in one probe.
template <typename M, typename Key, typename Value>
concept HasFastExtract = requires(M m, const Key& k) {
    { m.extract_ex(k) } -> std::same_as<ModifyResult<Value>>;
};

}  // namespace hm

#endif  // HM_SUBMAP_CONCEPTS_HPP