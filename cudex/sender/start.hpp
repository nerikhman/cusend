#pragma once

#include "../detail/prologue.hpp"

#include <utility>
#include "../detail/type_traits/is_detected.hpp"
#include "../detail/static_const.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
using start_member_function_t = decltype(std::declval<T>().start());

template<class T>
using has_start_member_function = is_detected<start_member_function_t, T>;


template<class T>
using start_free_function_t = decltype(start(std::declval<T>()));

template<class T>
using has_start_free_function = is_detected<start_free_function_t, T>;


// this is the type of start
struct dispatch_start
{
  CUDEX_EXEC_CHECK_DISABLE
  template<class T, 
           CUDEX_REQUIRES(has_start_member_function<T&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(T&& arg) const ->
    decltype(std::forward<T>(arg).start())
  {
    return std::forward<T>(arg).start();
  }

  CUDEX_EXEC_CHECK_DISABLE
  template<class T,
           CUDEX_REQUIRES(!has_start_member_function<T&&>::value and
                          has_start_free_function<T&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(T&& arg) const ->
    decltype(start(std::forward<T>(arg)))
  {
    return start(std::forward<T>(arg));
  }
};


} // end detail


namespace
{


// define the start customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& start = detail::static_const<detail::dispatch_start>::value;
#else
const __device__ detail::dispatch_start start;
#endif


} // end anonymous namespace


template<class T>
using start_t = decltype(CUDEX_NAMESPACE::start(std::declval<T>()));


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

