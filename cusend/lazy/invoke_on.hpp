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

#include "../detail/prologue.hpp"

#include <utility>
#include "../detail/static_const.hpp"
#include "../detail/type_traits/is_detected.hpp"
#include "detail/default_invoke_on.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S, class F, class... Args>
using invoke_on_member_function_t = decltype(std::declval<S>().invoke_on(std::declval<F>(), std::declval<Args>()...));

template<class S, class F, class... Args>
using has_invoke_on_member_function = is_detected<invoke_on_member_function_t, S, F, Args...>;


template<class S, class F, class... Args>
using invoke_on_free_function_t = decltype(invoke_on(std::declval<S>(), std::declval<F>(), std::declval<Args>()...));

template<class S, class F, class... Args>
using has_invoke_on_free_function = is_detected<invoke_on_free_function_t, S, F, Args...>;


// this is the type of the invoke_on CPO
struct dispatch_invoke_on
{
  CUSEND_EXEC_CHECK_DISABLE
  template<class S, class F, class... Args,
           CUSEND_REQUIRES(has_invoke_on_member_function<S&&,F&&,Args&&...>::value)
          >
  CUSEND_ANNOTATION
  constexpr invoke_on_member_function_t<S&&,F&&,Args&&...>
    operator()(S&& scheduler, F&& f, Args&&... args) const
  {
    return std::forward<S>(scheduler).invoke_on(std::forward<F>(f), std::forward<Args>(args)...);
  }

  CUSEND_EXEC_CHECK_DISABLE
  template<class S, class F, class... Args,
           CUSEND_REQUIRES(!has_invoke_on_member_function<S&&,F&&,Args&&...>::value),
           CUSEND_REQUIRES(has_invoke_on_free_function<S&&,F&&,Args&&...>::value)
          >
  CUSEND_ANNOTATION
  constexpr invoke_on_free_function_t<S&&,F&&,Args&&...>
    operator()(S&& scheduler, F&& f, Args&&... args) const
  {
    return invoke_on(std::forward<S>(scheduler), std::forward<F>(f), std::forward<Args>(args)...);
  }

  CUSEND_EXEC_CHECK_DISABLE
  template<class S, class F, class... Args,
           CUSEND_REQUIRES(!has_invoke_on_member_function<S&&,F&&,Args&&...>::value),
           CUSEND_REQUIRES(!has_invoke_on_free_function<S&&,F&&,Args&&...>::value),
           CUSEND_REQUIRES(is_detected<default_invoke_on_t,S&&,F&&,Args&&...>::value)
          >
  CUSEND_ANNOTATION
  constexpr default_invoke_on_t<S&&,F&&,Args&&...>
    operator()(S&& scheduler, F&& f, Args&&... args) const
  {
    return detail::default_invoke_on(std::forward<S>(scheduler), std::forward<F>(f), std::forward<Args>(args)...);
  }
};


} // end detail


namespace
{


// define the invoke_on customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& invoke_on = detail::static_const<detail::dispatch_invoke_on>::value;
#else
const __device__ detail::dispatch_invoke_on invoke_on;
#endif


} // end anonymous namespace


template<class S, class F, class... Args>
using invoke_on_t = decltype(invoke_on(std::declval<S>(), std::declval<F>(), std::declval<Args>()...));


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

