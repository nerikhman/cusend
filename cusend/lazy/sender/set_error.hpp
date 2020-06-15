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


template<class R, class E>
using set_error_member_t = decltype(std::declval<R>().set_error(std::declval<E>()));


template<class R, class E>
using has_set_error_member = is_detected<set_error_member_t, R, E>;


template<class R, class E>
using set_error_free_function_t = decltype(set_error(std::declval<R>(), std::declval<E>()));


template<class R, class E>
using has_set_error_free_function = is_detected<set_error_free_function_t, R, E>;


// this is the type of set_error
struct dispatch_set_error
{
  CUSEND_EXEC_CHECK_DISABLE
  template<class R, class E,
           CUSEND_REQUIRES(has_set_error_member<R&&,E&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr auto operator()(R&& r, E&& e) const ->
    decltype(std::forward<R>(r).set_error(std::forward<E>(e)))
  {
    return std::forward<R>(r).set_error(std::forward<E>(e));
  }

  CUSEND_EXEC_CHECK_DISABLE
  template<class R, class E,
           CUSEND_REQUIRES(!has_set_error_member<R&&,E&&>::value and
                          has_set_error_free_function<R&&,E&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr auto operator()(R&& r, E&& e) const ->
    decltype(set_error(std::forward<R>(r), std::forward<E>(e)))
  {
    return set_error(std::forward<R>(r), std::forward<E>(e));
  }
};


} // end detail


namespace
{


// define the set_error customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& set_error = detail::static_const<detail::dispatch_set_error>::value;
#else
const __device__ detail::dispatch_set_error set_error;
#endif


} // end anonymous namespace


template<class T, class E>
using set_error_t = decltype(CUSEND_NAMESPACE::set_error(std::declval<T>(), std::declval<E>()));


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

