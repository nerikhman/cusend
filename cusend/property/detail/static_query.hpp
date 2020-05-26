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

#include <type_traits>
#include "../../detail/type_traits/is_detected.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


// c++11 property implementations cannot define member variable templates
// so, support a member function template named Prop::static_query<T>()

template<class P, class T>
using static_query_member_function_t = decltype(P::template static_query<T>());

template<class P, class T>
using has_static_query_member_function = is_detected<static_query_member_function_t, P, T>;


template<class P, class T>
using static_query_member_variable_t = decltype(P::template static_query_v<T>);

template<class P, class T>
using has_static_query_member_variable = is_detected<static_query_member_variable_t, P, T>;


template<class T, class P, class Enable = void>
class static_query;


// the template parameters T & P of static_query are ordered to match those of query()
template<class T, class P>
class static_query<T, P,
  typename std::enable_if<
    has_static_query_member_function<P,T>::value or
    has_static_query_member_variable<P,T>::value
  >::type
>
{
  private:
    template<class T_, class P_,
             CUSEND_REQUIRES(has_static_query_member_function<P_,T_>::value)
            >
    CUSEND_ANNOTATION
    constexpr static static_query_member_function_t<P_,T_> test()
    {
      return P_::template static_query<T_>();
    }
    
    
    template<class T_, class P_,
             CUSEND_REQUIRES(!has_static_query_member_function<P_,T_>::value),
             CUSEND_REQUIRES(has_static_query_member_variable<P_,T_>::value)
            >
    CUSEND_ANNOTATION
    constexpr static static_query_member_variable_t<P_,T_> test()
    {
      return P_::template static_query_v<T_>;
    }

  public:
    static constexpr decltype(auto) value = static_query::test<T,P>();
    using type = decltype(value);
};


template<class T, class P>
using static_query_t = typename static_query<T,P>::type;


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../detail/epilogue.hpp"

