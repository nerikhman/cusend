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
#include "../type_traits/is_detected.hpp"
#include "default_on.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S, class E>
using on_member_function_t = decltype(std::declval<S>().on(std::declval<E>()));

template<class S, class E>
using has_on_member_function = is_detected<on_member_function_t, S, E>;


template<class S, class E>
using on_free_function_t = decltype(on(std::declval<S>(), std::declval<E>()));

template<class S, class E>
using has_on_free_function = is_detected<on_free_function_t, S, E>;


// this is the type of the on CPO
struct dispatch_on
{
  // dispatch case 1: sender.on(ex) exists
  template<class Sender, class Executor,
           CUSEND_REQUIRES(has_on_member_function<Sender&&,const Executor&>::value)
          >
  CUSEND_ANNOTATION
  constexpr on_member_function_t<Sender&&,const Executor&>
    operator()(Sender&& sender, const Executor& ex) const
  {
    return std::forward<Sender>(sender).on(ex);
  }
  
  
  // dispatch case 1: sender.on(ex) does not exist
  //                  on(sender, ex) does exist
  template<class Sender, class Executor,
           CUSEND_REQUIRES(!has_on_member_function<Sender&&,const Executor&>::value),
           CUSEND_REQUIRES(has_on_free_function<Sender&&,const Executor&>::value)
          >
  CUSEND_ANNOTATION
  constexpr on_free_function_t<Sender&&,const Executor&>
    operator()(Sender&& sender, const Executor& ex) const
  {
    return on(std::forward<Sender>(sender), ex);
  }
  
  
  // dispatch case 2: sender.on(ex) does not exist
  //                  on(sender, ex) does not exist
  template<class Sender, class Executor,
           CUSEND_REQUIRES(!has_on_member_function<Sender&&,const Executor&>::value),
           CUSEND_REQUIRES(!has_on_free_function<Sender&&,const Executor&>::value),
           CUSEND_REQUIRES(is_detected<default_on_t,Sender&&,const Executor&>::value)
          >
  CUSEND_ANNOTATION
  constexpr default_on_t<Sender&&,const Executor&>
    operator()(Sender&& sender, const Executor& ex) const
  {
    return detail::default_on(std::forward<Sender>(sender), ex);
  }
};


namespace
{


// define the on customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& on = detail::static_const<detail::dispatch_on>::value;
#else
const __device__ detail::dispatch_on on;
#endif


} // end anonymous namespace


template<class S, class E>
using on_t = decltype(detail::on(std::declval<S>(), std::declval<E>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../epilogue.hpp"

