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
#include "../detail/type_traits/is_detected.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class P, class T>
using is_applicable_property_member_variable_t = decltype(P::template is_applicable_property_v<T>);


// c++11 property implementations cannot define member variable templates
// so, support a member function template named Prop::is_applicable_property<T>()
template<class P, class T>
using is_applicable_property_member_function_t = decltype(P::template is_applicable_property<T>());


template<class T, class P>
struct is_applicable_property_impl
{
  // first, check for a member function named P::is_applicable_property<T>()
  template<class T_, class P_,
           CUSEND_REQUIRES(is_detected<is_applicable_property_member_function_t,P_,T_>::value)
          >
  constexpr static bool test(int)
  {
    return P::template is_applicable_property<T>();
  }

  // next, check for a member variable named P::is_applicable_property_v<T>
  template<class T_, class P_,
           CUSEND_REQUIRES(!is_detected<is_applicable_property_member_function_t,P_,T_>::value),
           CUSEND_REQUIRES(is_detected<is_applicable_property_member_variable_t,P_,T_>::value)
          >
  constexpr static bool test(int)
  {
    return P::template is_applicable_property_v<T>;
  }
  
  // finally, return false
  template<class T_, class P_,
           CUSEND_REQUIRES(!is_detected<is_applicable_property_member_function_t,P_,T_>::value),
           CUSEND_REQUIRES(!is_detected<is_applicable_property_member_variable_t,P_,T_>::value)
          >
  constexpr static bool test(...)
  {
    return false;
  }

  constexpr static bool value = test<T,P>(0);
};


} // end detail


template<class T, class P>
using is_applicable_property = std::integral_constant<
  bool,
  detail::is_applicable_property_impl<T,P>::value
>;


#ifdef __cpp_variable_templates
template<class T, class P>
constexpr static bool is_applicable_property_v = is_applicable_property<T,P>::value;
#endif


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

