// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "../../detail/prologue.hpp"

#include <type_traits>
#include "../../detail/type_traits/conjunction.hpp"
#include "../../detail/type_traits/remove_cvref.hpp"
#include "../../execution/executor/is_executor_of.hpp"
#include "../detail/detail/receiver_as_invocable.hpp"
#include "sender_base.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


template<class S>
struct sender_traits;


namespace detail
{


template<class T>
struct has_value_types
{
  template<class...>
  struct template_template_parameter {};

  template<class U = T,
           class = typename U::template value_types<template_template_parameter, template_template_parameter>
          >
  constexpr static bool test(int)
  {
    return true;
  }

  template<class>
  constexpr static bool test(...)
  {
    return false;
  }

  constexpr static bool value = test<T>(0);
};


template<class T>
struct has_error_types
{
  template<class...>
  struct template_template_parameter {};

  template<class U = T,
           class = typename U::template error_types<template_template_parameter>
          >
  constexpr static bool test(int)
  {
    return true;
  }

  template<class>
  constexpr static bool test(...)
  {
    return false;
  }

  constexpr static bool value = test<T>(0);
};


template<class T>
struct has_sends_done
{
  template<class U = T,
           bool = U::sends_done
          >
  constexpr static bool test(int)
  {
    return true;
  }

  template<class>
  constexpr static bool test(...)
  {
    return false;
  }

  constexpr static bool value = test<T>(0);
};


template<class T>
using has_sender_types = conjunction<has_value_types<T>, has_error_types<T>, has_sends_done<T>>;


template<class S, class Enable = void>
struct sender_traits_base
{
  using __unspecialized = void;
};


// If S has sender types, use them
template<class S>
struct sender_traits_base<
  S,
  typename std::enable_if<
    has_sender_types<S>::value
  >::type
>
{
  template<template<class...> class Tuple, template<class...> class Variant>
  using value_types = typename S::template value_types<Tuple, Variant>;

  template<template<class...> class Variant>
  using error_types = typename S::template error_types<Variant>;

  static constexpr bool sends_done = S::sends_done;
};


// Otherwise, if executor-of-impl<S,as-invocable<void-receiver, S>> is true, then sender-traits-base is equivalent to...

struct void_receiver
{
  void set_value() noexcept;
  void set_error(std::exception_ptr) noexcept;
  void set_done() noexcept;
};


template<class S>
struct sender_traits_base<
  S,
  typename std::enable_if<
    !has_sender_types<S>::value and
    execution::is_executor_of<S, receiver_as_invocable<void_receiver>>::value
  >::type
>
{
  template<template<class...> class Tuple, template<class...> class Variant>
  using value_types = Variant<Tuple<>>;

  template<template<class...> class Variant>
  using error_types = Variant<std::exception_ptr>;

  static constexpr bool sends_done = true;
};


// Otherwise, if S does not have sender types, and S is derived from sender_base
template<class Derived, class Base>
using is_derived_from = std::integral_constant<
  bool,
  std::is_base_of<Base,Derived>::value and
  std::is_convertible<const volatile Derived*, const volatile Base*>::value
>;

template<class S>
struct sender_traits_base<
  S,
  typename std::enable_if<
    !has_sender_types<S>::value and
    !execution::is_executor_of<S, receiver_as_invocable<void_receiver>>::value and
    is_derived_from<S, sender_base>::value
  >::type
>
{
  // empty
};


} // end detail


template<class S>
struct sender_traits : detail::sender_traits_base<detail::remove_cvref_t<S>> {};


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../detail/epilogue.hpp"

