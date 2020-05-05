#pragma once

#include "../detail/prologue.hpp"

#include <utility>
#include "../detail/static_const.hpp"
#include "../detail/type_traits/is_detected.hpp"
#include "detail/default_submit.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S, class R>
using submit_member_function_t = decltype(std::declval<S>().submit(std::declval<R>()));

template<class S, class R>
using has_submit_member_function = is_detected<submit_member_function_t, S, R>;


template<class S, class R>
using submit_free_function_t = decltype(submit(std::declval<S>(), std::declval<R>()));

template<class S, class R>
using has_submit_free_function = is_detected<submit_free_function_t, S, R>;


// this is the type of submit
struct dispatch_submit
{
  CUDEX_EXEC_CHECK_DISABLE
  template<class S, class R,
           CUDEX_REQUIRES(has_submit_member_function<S&&,R&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(S&& s, R&& r) const ->
    decltype(std::forward<S>(s).submit(std::forward<R>(r)))
  {
    return std::forward<S>(s).submit(std::forward<R>(r));
  }

  CUDEX_EXEC_CHECK_DISABLE
  template<class S, class R,
           CUDEX_REQUIRES(!has_submit_member_function<S&&,R&&>::value and
                          has_submit_free_function<S&&,R&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(S&& s, R&& r) const ->
    decltype(submit(std::forward<S>(s), std::forward<R>(r)))
  {
    return submit(std::forward<S>(s), std::forward<R>(r));
  }

  CUDEX_EXEC_CHECK_DISABLE
  template<class S, class R,
           CUDEX_REQUIRES(!has_submit_member_function<S&&,R&&>::value and
                          !has_submit_free_function<S&&,R&&>::value and
                          is_detected<default_submit_t, S&&, R&&>::value)
          >
  CUDEX_ANNOTATION
  void operator()(S&& s, R&& r) const
  {
    detail::default_submit(std::forward<S>(s), std::forward<R>(r));
  }
};


} // end detail


namespace
{


// define the submit customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& submit = detail::static_const<detail::dispatch_submit>::value;
#else
const __device__ detail::dispatch_submit submit;
#endif


} // end anonymous namespace


template<class T>
using submit_t = decltype(CUDEX_NAMESPACE::submit(std::declval<T>()));


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

