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

#include <utility>
#include "../../detail/static_const.hpp"
#include "../../detail/type_traits/is_detected.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


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
  CUSEND_EXEC_CHECK_DISABLE
  template<class S, class R,
           CUSEND_REQUIRES(has_connect_member_function<S&&,R&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr auto operator()(S&& s, R&& r) const ->
    decltype(std::forward<S>(s).connect(std::forward<R>(r)))
  {
    return std::forward<S>(s).connect(std::forward<R>(r));
  }

  CUSEND_EXEC_CHECK_DISABLE
  template<class S, class R,
           CUSEND_REQUIRES(!has_connect_member_function<S&&,R&&>::value and
                          has_connect_free_function<S&&,R&&>::value)
          >
  CUSEND_ANNOTATION
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
using connect_t = decltype(CUSEND_NAMESPACE::connect(std::declval<S>(), std::declval<R>()));


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

