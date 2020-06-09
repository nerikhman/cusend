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
#include "default_just_on.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S, class... Types>
using just_on_member_function_t = decltype(std::declval<S>().just_on(std::declval<Types>()...));

template<class S, class... Types>
using has_just_on_member_function = is_detected<just_on_member_function_t, S, Types...>;


template<class S, class... Types>
using just_on_free_function_t = decltype(just_on(std::declval<S>(), std::declval<Types>()...));

template<class S, class... Types>
using has_just_on_free_function = is_detected<just_on_free_function_t, S, Types...>;


// this is the type of the just_on CPO
struct dispatch_just_on
{
  CUSEND_EXEC_CHECK_DISABLE
  template<class S, class... Types,
           CUSEND_REQUIRES(has_just_on_member_function<S&&,Types&&...>::value)
          >
  CUSEND_ANNOTATION
  constexpr just_on_member_function_t<S&&,Types&&...>
    operator()(S&& scheduler, Types&&... values) const
  {
    return std::forward<S>(scheduler).just_on(std::forward<Types>(values)...);
  }

  CUSEND_EXEC_CHECK_DISABLE
  template<class S, class... Types,
           CUSEND_REQUIRES(!has_just_on_member_function<S&&,Types&&...>::value),
           CUSEND_REQUIRES(has_just_on_free_function<S&&,Types&&...>::value)
          >
  CUSEND_ANNOTATION
  constexpr just_on_free_function_t<S&&,Types&&...>
    operator()(S&& scheduler, Types&&... values) const
  {
    return just_on(std::forward<S>(scheduler), std::forward<Types>(values)...);
  }

  CUSEND_EXEC_CHECK_DISABLE
  template<class S, class... Types,
           CUSEND_REQUIRES(!has_just_on_member_function<S&&,Types&&...>::value),
           CUSEND_REQUIRES(!has_just_on_free_function<S&&,Types&&...>::value),
           CUSEND_REQUIRES(is_detected<default_just_on_t,S&&,Types&&...>::value)
          >
  CUSEND_ANNOTATION
  constexpr default_just_on_t<S&&,Types&&...>
    operator()(S&& scheduler, Types&&... values) const
  {
    return detail::default_just_on(std::forward<S>(scheduler), std::forward<Types>(values)...);
  }
};


namespace
{


// define the just_on customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& just_on = detail::static_const<detail::dispatch_just_on>::value;
#else
const __device__ detail::dispatch_just_on just_on;
#endif


} // end anonymous namespace


template<class S, class... Types>
using just_on_t = decltype(detail::just_on(std::declval<S>(), std::declval<Types>()...));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../epilogue.hpp"

