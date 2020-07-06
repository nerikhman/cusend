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
#include "detail/default_as_scheduler.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E>
using as_scheduler_member_function_t = decltype(std::declval<E>().as_scheduler());

template<class E>
using has_as_scheduler_member_function = is_detected<as_scheduler_member_function_t, E>;


template<class E>
using as_scheduler_free_function_t = decltype(as_scheduler(std::declval<E>()));

template<class E>
using has_as_scheduler_free_function = is_detected<as_scheduler_free_function_t, E>;


// this is the type of the as_scheduler CPO
// XXX consider requiring that the result of these overloads satisfy is_scheduler
struct dispatch_as_scheduler
{
  CUSEND_EXEC_CHECK_DISABLE
  template<class Executor,
           CUSEND_REQUIRES(has_as_scheduler_member_function<Executor&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr as_scheduler_member_function_t<Executor&&>
    operator()(Executor&& executor) const
  {
    return std::forward<Executor>(executor).as_scheduler();
  }


  CUSEND_EXEC_CHECK_DISABLE
  template<class Executor,
           CUSEND_REQUIRES(!has_as_scheduler_member_function<Executor&&>::value),
           CUSEND_REQUIRES(has_as_scheduler_free_function<Executor&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr as_scheduler_free_function_t<Executor&&>
    operator()(Executor&& executor) const
  {
    return as_scheduler(std::forward<Executor>(executor));
  }


  CUSEND_EXEC_CHECK_DISABLE
  template<class Executor,
           CUSEND_REQUIRES(!has_as_scheduler_member_function<Executor&&>::value),
           CUSEND_REQUIRES(!has_as_scheduler_free_function<Executor&&>::value),
           CUSEND_REQUIRES(is_detected<default_as_scheduler_t,Executor&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr default_as_scheduler_t<Executor&&>
    operator()(Executor&& executor) const
  {
    return detail::default_as_scheduler(std::forward<Executor>(executor));
  }
};


} // end detail


namespace
{


// define the as_scheduler customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& as_scheduler = detail::static_const<detail::dispatch_as_scheduler>::value;
#else
const __device__ detail::dispatch_as_scheduler as_scheduler;
#endif


} // end anonymous namespace


template<class E>
using as_scheduler_t = decltype(as_scheduler(std::declval<E>()));


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

