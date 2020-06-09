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
#include "default_pack.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S>
using pack_member_function_t = decltype(std::declval<S>().pack());

template<class S>
using has_pack_member_function = is_detected<pack_member_function_t, S>;


template<class S>
using pack_free_function_t = decltype(pack(std::declval<S>()));

template<class S>
using has_pack_free_function = is_detected<pack_free_function_t, S>;


// this is the type of the pack CPO
struct dispatch_pack
{
  CUSEND_EXEC_CHECK_DISABLE
  template<class S,
           CUSEND_REQUIRES(has_pack_member_function<S&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr pack_member_function_t<S&&>
    operator()(S&& s) const
  {
    return std::forward<S>(s).pack();
  }

  CUSEND_EXEC_CHECK_DISABLE
  template<class S,
           CUSEND_REQUIRES(!has_pack_member_function<S&&>::value),
           CUSEND_REQUIRES(has_pack_free_function<S&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr pack_free_function_t<S&&>
    operator()(S&& s) const
  {
    return pack(std::forward<S>(s));
  }

  CUSEND_EXEC_CHECK_DISABLE
  template<class S,
           CUSEND_REQUIRES(!has_pack_member_function<S&&>::value),
           CUSEND_REQUIRES(!has_pack_free_function<S&&>::value),
           CUSEND_REQUIRES(is_detected<default_pack_t, S&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr default_pack_t<S&&>
    operator()(S&& s) const
  {
    return detail::default_pack(std::forward<S>(s));
  }
};


namespace
{


// define the pack customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& pack = detail::static_const<detail::dispatch_pack>::value;
#else
const __device__ detail::dispatch_pack pack;
#endif


} // end anonymous namespace


template<class S>
using pack_t = decltype(detail::pack(std::declval<S>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../epilogue.hpp"

