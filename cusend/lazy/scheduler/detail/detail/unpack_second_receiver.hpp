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

#include "../../../../detail/prologue.hpp"

#include <cstdint>
#include <tuple>
#include <utility>
#include "../../../../detail/tuple.hpp"
#include "../../../../detail/utility/index_sequence.hpp"
#include "../../../receiver/set_done.hpp"
#include "../../../receiver/set_error.hpp"
#include "../../../receiver/set_value.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


// unpack_second_receiver adapts an underlying receiver
// unpack_second_receiver::set_value calls set_value on an underlying receiver
// unpack_second_receiver::set_value expects its second value argument to be a detail::tuple
// it forwards the value in the first argument slot to set_value
// the elements of the tuple are forwarded separately as individual values
// any other values are forwarded along afterward
template<class Receiver>
class unpack_second_receiver
{
  private:
    Receiver receiver_;


    template<std::size_t... I, class R, class T1, class Tuple, class... Types>
    CUSEND_ANNOTATION
    static auto set_value_impl(index_sequence<I...>, R&& r, T1&& value1, Tuple&& value2, Types&&... values)
    {
      // the implementation of set_value calls set_value on the underlying receiver
      // it expects the second value argument to be a tuple
      // it forwards the first value in the first argument slot to set_value
      // the elements of the tuple are forwarded separately as individual values
      // any other values are forwarded along afterward
      return CUSEND_NAMESPACE::set_value(std::forward<R>(r), std::forward<T1>(value1), detail::get<I>(std::forward<Tuple>(value2))..., std::forward<Types>(values)...);
    }


  public:
    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    unpack_second_receiver(Receiver&& receiver)
      : receiver_{std::move(receiver)}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    unpack_second_receiver(const unpack_second_receiver&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    unpack_second_receiver(unpack_second_receiver&&) = default;


    template<class T1, class Tuple, class... Types>
    CUSEND_ANNOTATION
    void set_value(T1&& value1, Tuple&& value2, Types&&... values) &&
    {
      constexpr std::size_t N = std::tuple_size<remove_cvref_t<Tuple>>::value;

      this->set_value_impl(make_index_sequence<N>{}, std::move(receiver_), std::forward<T1>(value1), std::forward<Tuple>(value2), std::forward<Types>(values)...);
    }


    template<class T1, class Tuple, class... Types>
    CUSEND_ANNOTATION
    void set_value(T1&& value1, Tuple&& value2, Types&&... values) &
    {
      constexpr std::size_t N = std::tuple_size<remove_cvref_t<Tuple>>::value;

      this->set_value_impl(make_index_sequence<N>{}, receiver_, std::forward<T1>(value1), std::forward<Tuple>(value2), std::forward<Types>(values)...);
    }


    template<class T1, class Tuple, class... Types>
    CUSEND_ANNOTATION
    void set_value(T1&& value1, Tuple&& value2, Types&&... values) const &
    {
      constexpr std::size_t N = std::tuple_size<remove_cvref_t<Tuple>>::value;

      this->set_value_impl(make_index_sequence<N>{}, receiver_, std::forward<T1>(value1), std::forward<Tuple>(value2), std::forward<Types>(values)...);
    }


    template<class E>
    CUSEND_ANNOTATION
    void set_error(E&& e) &&
    {
      CUSEND_NAMESPACE::set_error(std::move(receiver_), std::forward<E>(e));
    }


    CUSEND_ANNOTATION
    void set_done() &&
    {
      CUSEND_NAMESPACE::set_done(std::move(receiver_));
    }
};


template<class Receiver>
CUSEND_ANNOTATION
unpack_second_receiver<remove_cvref_t<Receiver>> make_unpack_second_receiver(Receiver&& r)
{
  return {std::forward<Receiver>(r)};
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../../../detail/epilogue.hpp"

