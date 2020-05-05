#pragma once

#include "../detail/prologue.hpp"

#include <utility>
#include "../detail/static_const.hpp"
#include "../detail/type_traits/is_detected.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class R, class... Args>
using set_value_member_function_t = decltype(std::declval<R>().set_value(std::declval<Args>()...));

template<class R, class... Args>
using has_set_value_member_function = is_detected<set_value_member_function_t, R, Args...>;


template<class R, class... Args>
using set_value_free_function_t = decltype(set_value(std::declval<R>(), std::declval<Args>()...));

template<class R, class... Args>
using has_set_value_free_function = is_detected<set_value_free_function_t, R, Args...>;


// this is the type of set_value
struct dispatch_set_value
{
  CUDEX_EXEC_CHECK_DISABLE
  template<class R, class... Args,
           CUDEX_REQUIRES(has_set_value_member_function<R&&,Args&&...>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(R&& r, Args&&... args) const ->
    decltype(std::forward<R>(r).set_value(std::forward<Args>(args)...))
  {
    return std::forward<R>(r).set_value(std::forward<Args>(args)...);
  }

  CUDEX_EXEC_CHECK_DISABLE
  template<class R, class... Args,
           CUDEX_REQUIRES(!has_set_value_member_function<R&&,Args&&...>::value and
                          has_set_value_free_function<R&&,Args&&...>::value)
           >
  CUDEX_ANNOTATION
  constexpr auto operator()(R&& r, Args&&... args) const ->
    decltype(set_value(std::forward<R>(r), std::forward<Args>(args)...))
  {
    return set_value(std::forward<R>(r), std::forward<Args>(args)...);
  }
};


} // end detail



namespace
{


// define the set_value customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& set_value = detail::static_const<detail::dispatch_set_value>::value;
#else
const __device__ detail::dispatch_set_value set_value;
#endif


} // end anonymous namespace


template<class T, class... Args>
using set_value_t = decltype(CUDEX_NAMESPACE::set_value(std::declval<T>(), std::declval<Args>()...));


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

