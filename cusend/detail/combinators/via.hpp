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

#include "../prologue.hpp"

#include <utility>
#include "../static_const.hpp"
#include "../type_traits/is_detected.hpp"
#include "default_via.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S, class E>
using via_member_function_t = decltype(std::declval<S>().via(std::declval<E>()));

template<class S, class E>
using has_via_member_function = is_detected<via_member_function_t, S, E>;


template<class S, class E>
using via_free_function_t = decltype(via(std::declval<S>(), std::declval<E>()));

template<class S, class E>
using has_via_free_function = is_detected<via_free_function_t, S, E>;


// this is the type of the via CPO
struct dispatch_via
{
  CUSEND_EXEC_CHECK_DISABLE
  template<class S, class E,
           CUSEND_REQUIRES(has_via_member_function<S&&,E&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr via_member_function_t<S&&,E&&>
    operator()(S&& predecessor, E&& ex) const
  {
    return std::forward<S>(predecessor).via(std::forward<E>(ex));
  }


  CUSEND_EXEC_CHECK_DISABLE
  template<class S, class E,
           CUSEND_REQUIRES(!has_via_member_function<S&&,E&&>::value),
           CUSEND_REQUIRES(has_via_free_function<S&&,E&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr via_free_function_t<E&&,S&&>
    operator()(S&& predecessor, E&& ex) const
  {
    return via(std::forward<S>(predecessor), std::forward<E>(ex));
  }


  CUSEND_EXEC_CHECK_DISABLE
  template<class S, class E,
           CUSEND_REQUIRES(!has_via_member_function<S&&,E&&>::value),
           CUSEND_REQUIRES(!has_via_free_function<S&&,E&&>::value),
           CUSEND_REQUIRES(is_detected<default_via_t,S&&,E&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr default_via_t<S&&,E&&>
    operator()(S&& predecessor, E&& ex) const
  {
    return detail::default_via(std::forward<S>(predecessor), std::forward<E>(ex));
  }
};


namespace
{


// define the via customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& via = detail::static_const<detail::dispatch_via>::value;
#else
const __device__ detail::dispatch_via via;
#endif


} // end anonymous namespace


template<class S, class E>
using via_t = decltype(detail::via(std::declval<S>(), std::declval<E>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../epilogue.hpp"

