#pragma once

#include "../detail/prologue.hpp"

#include <utility>
#include "../detail/static_const.hpp"
#include "../detail/type_traits/is_detected.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S, class R>
using connect_member_function_t = decltype(std::declval<S>().connect(std::declval<R>()));

template<class S, class R>
using has_connect_member_function = is_detected<connect_member_function_t, S, R>;


template<class S, class R>
using connect_free_function_t = decltype(connect(std::declval<S>(), std::declval<R>()));

template<class S, class R>
using has_connect_free_function = is_detected<connect_free_function_t, S, R>;


// this is the type of connect
struct dispatch_connect
{
  CUDEX_EXEC_CHECK_DISABLE
  template<class S, class R,
           CUDEX_REQUIRES(has_connect_member_function<S&&,R&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(S&& s, R&& r) const ->
    decltype(std::forward<S>(s).connect(std::forward<R>(r)))
  {
    return std::forward<S>(s).connect(std::forward<R>(r));
  }

  CUDEX_EXEC_CHECK_DISABLE
  template<class S, class R,
           CUDEX_REQUIRES(!has_connect_member_function<S&&,R&&>::value and
                          has_connect_free_function<S&&,R&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(S&& s, R&& r) const ->
    decltype(connect(std::forward<S>(s), std::forward<R>(r)))
  {
    return connect(std::forward<S>(s), std::forward<R>(r));
  }
};


} // end detail


namespace
{


// define the connect customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& connect = detail::static_const<detail::dispatch_connect>::value;
#else
const __device__ detail::dispatch_connect connect;
#endif


} // end anonymous namespace


template<class S, class R>
using connect_t = decltype(CUDEX_NAMESPACE::connect(std::declval<S>(), std::declval<R>()));


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

