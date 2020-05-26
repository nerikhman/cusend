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

#include <type_traits>
#include "../detail/static_const.hpp"
#include "../detail/type_traits/is_detected.hpp"
#include "../detail/type_traits/remove_cvref.hpp"
#include "detail/static_query.hpp"
#include "is_applicable_property.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T, class P>
using query_member_function_t = decltype(std::declval<T>().query(std::declval<P>()));

template<class T, class P>
using has_query_member_function = is_detected<query_member_function_t, T, P>;


template<class T, class P>
using query_free_function_t = decltype(query(std::declval<T>(), std::declval<P>()));

template<class T, class P>
using has_query_free_function = is_detected<query_free_function_t, T, P>;


// this is the type of the query CPO
struct dispatch_query
{
  // first, try a static_query
  template<class T, class P,
           CUSEND_REQUIRES(is_applicable_property<typename std::decay<T>::type, typename std::decay<P>::type>::value),
           CUSEND_REQUIRES(is_detected<static_query_t, typename std::decay<T>::type, typename std::decay<P>::type>::value)
          >
  CUSEND_ANNOTATION
  constexpr static_query_t<typename std::decay<T>::type, typename std::decay<P>::type>
    operator()(T&&, P&&) const
  {
    return detail::static_query<typename std::decay<T>::type, typename std::decay<P>::type>();
  }

  // next, try a member function .query()
  CUSEND_EXEC_CHECK_DISABLE
  template<class T, class P,
           CUSEND_REQUIRES(is_applicable_property<typename std::decay<T>::type, typename std::decay<P>::type>::value),
           CUSEND_REQUIRES(!is_detected<static_query_t, typename std::decay<T>::type, typename std::decay<P>::type>::value),
           CUSEND_REQUIRES(has_query_member_function<T&&,P&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr query_member_function_t<T&&,P&&>
    operator()(T&& arg, P&& property) const
  {
    return std::forward<T>(arg).query(std::forward<P>(property));
  }

  // finally, try a free function query()
  CUSEND_EXEC_CHECK_DISABLE
  template<class T, class P,
           CUSEND_REQUIRES(is_applicable_property<typename std::decay<T>::type, typename std::decay<P>::type>::value),
           CUSEND_REQUIRES(!is_detected<static_query_t, typename std::decay<T>::type, typename std::decay<P>::type>::value),
           CUSEND_REQUIRES(!has_query_member_function<T&&,P&&>::value),
           CUSEND_REQUIRES(has_query_free_function<T&&,P&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr query_free_function_t<T&&,P&&>
    operator()(T&& arg, P&& property) const
  {
    return query(std::forward<T>(arg), std::forward<P>(property));
  }
};


} // end detail


namespace
{


// define the query customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& query = detail::static_const<detail::dispatch_query>::value;
#else
const __device__ detail::dispatch_query query;
#endif


} // end anonymous namespace


template<class T, class P>
using query_t = decltype(CUSEND_NAMESPACE::query(std::declval<T>(), std::declval<P>()));


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

