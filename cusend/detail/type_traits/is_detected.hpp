// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "../prologue.hpp"

#include <type_traits>


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{
namespace is_detected_detail
{


template<class...> 
using void_t = void; 
 
struct nonesuch 
{ 
  nonesuch() = delete; 
  ~nonesuch() = delete; 
  nonesuch(const nonesuch&) = delete; 
  void operator=(const nonesuch&) = delete; 
}; 
 
 
template<class Default, class AlwaysVoid,
         template<class...> class Op, class... Args> 
struct detector 
{ 
  using value_t = std::false_type; 
  using type = Default; 
}; 
 
 
template<class Default, template<class...> class Op, class... Args> 
struct detector<Default, void_t<Op<Args...>>, Op, Args...> 
{ 
  using value_t = std::true_type; 
  using type = Op<Args...>; 
}; 


} // end is_detected_detail
 
 
template<template<class...> class Op, class... Args> 
using is_detected = typename is_detected_detail::detector<is_detected_detail::nonesuch, void, Op, Args...>::value_t; 
 
template<template<class...> class Op, class... Args> 
using detected_t = typename is_detected_detail::detector<is_detected_detail::nonesuch, void, Op, Args...>::type; 
 
template<class Default, template<class...> class Op, class... Args> 
using detected_or = is_detected_detail::detector<Default, void, Op, Args...>; 
 
template<class Default, template<class...> class Op, class... Args> 
using detected_or_t = typename detected_or<Default,Op,Args...>::type; 


template<class Expected, template<class...> class Op, class... Args>
using is_detected_exact = std::is_same<Expected, detected_t<Op,Args...>>;

template<class To, template<class...> class Op, class... Args>
using is_detected_convertible = std::is_convertible<detected_t<Op, Args...>, To>;


#if defined(__cpp_variable_templates)

template<class Expected, template<class...> class Op, class... Args>
constexpr bool is_detected_exact_v = is_detected_exact<Expected, Op, Args...>::value;

template<class To, template<class...> class Op, class... Args>
constexpr bool is_detected_convertible_v = is_detected_convertible<To, Op, Args...>::value;

#endif


namespace is_detected_detail
{


template<template<class> class Predicate, template<class...> class Op, class... Args>
struct is_detected_and_impl
{
  template<int,
           CUSEND_REQUIRES(is_detected<Op,Args...>::value)
          >
  constexpr static bool test(int)
  {
    return Predicate<Op<Args...>>::value;
  }

  template<int>
  constexpr static bool test(...)
  {
    return false;
  }

  using type = std::integral_constant<bool, test<0>(0)>;
};


} // end is_detected_detail


template<template<class> class Predicate, template<class...> class Op, class... Args>
using is_detected_and = typename is_detected_detail::is_detected_and_impl<Predicate, Op, Args...>::type;


#if defined(__cpp_variable_templates)

template<template<class> class Predicate, template<class...> class Op, class... Args>
constexpr bool is_detected_and_v = is_detected_and<Predicate, Op, Args...>::value;

#endif


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

