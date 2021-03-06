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

#include "../../../detail/prologue.hpp"

#include <cstdint>
#include <tuple>
#include "../../../detail/tuple.hpp"
#include "../../../detail/type_traits/remove_cvref.hpp"
#include "../../../detail/utility/index_sequence.hpp"
#include "../../connect.hpp"
#include "../../is_sender_to.hpp"
#include "../../receiver/discard_receiver.hpp"
#include "../../receiver/is_receiver_of.hpp"
#include "../../sender/is_typed_sender.hpp"
#include "../../sender/sender_base.hpp"
#include "detail/type_list.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<std::size_t I, class T>
using get_t = decltype(detail::get<I>(std::declval<T>()));


template<class R, class Tuple>
struct is_receiver_of_unpacked_tuple_impl
{
  template<std::size_t... I>
  static constexpr bool test_impl(index_sequence<I...>)
  {
    return is_receiver_of<R, get_t<I,Tuple>...>::value;
  }

  template<class T, std::size_t N = std::tuple_size<remove_cvref_t<T>>::value>
  static constexpr bool test(int)
  {
    return test_impl(make_index_sequence<N>{});
  }

  template<class>
  static constexpr bool test(...)
  {
    return false;
  }

  constexpr static bool value = test<Tuple>(0);
};


template<class R, class Tuple>
using is_receiver_of_unpacked_tuple = std::integral_constant<
  bool,
  is_receiver_of_unpacked_tuple_impl<R,Tuple>::value
>;


template<class Receiver>
class unpack_receiver
{
  private:
    Receiver receiver_;

    template<class Tuple, std::size_t... I>
    CUSEND_ANNOTATION
    void set_value_impl(Tuple&& values, index_sequence<I...>) &&
    {
      CUSEND_NAMESPACE::set_value(std::move(receiver_), detail::get<I>(std::forward<Tuple>(values))...);
    }

  public:
    CUSEND_EXEC_CHECK_DISABLE
    unpack_receiver(const unpack_receiver&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    unpack_receiver(unpack_receiver&&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    template<class R,
             CUSEND_REQUIRES(std::is_constructible<Receiver,R&&>::value)
            >
    CUSEND_ANNOTATION
    unpack_receiver(R&& receiver)
      : receiver_(std::forward<R>(receiver))
    {}

    template<class Tuple,
             CUSEND_REQUIRES(is_receiver_of_unpacked_tuple<Receiver&&, Tuple&&>::value)
            >
    CUSEND_ANNOTATION
    void set_value(Tuple&& values) &&
    {
      std::move(*this).set_value_impl(std::forward<Tuple>(values), make_index_sequence<std::tuple_size<remove_cvref_t<Tuple>>::value>{});
    }

    template<class Error,
             CUSEND_REQUIRES(is_receiver<Receiver, Error>::value)
            >
    CUSEND_ANNOTATION
    void set_error(Error&& error) && noexcept
    {
      CUSEND_NAMESPACE::set_error(std::move(receiver_), std::forward<Error>(error));
    }

    CUSEND_ANNOTATION
    void set_done() && noexcept
    {
      CUSEND_NAMESPACE::set_done(std::move(receiver_));
    }
};


template<class Receiver>
CUSEND_ANNOTATION
unpack_receiver<remove_cvref_t<Receiver>> make_unpack_receiver(Receiver&& receiver)
{
  return {std::forward<Receiver>(receiver)};
}


// this metafunction takes a tuple-like type,
// and "unpacks" the types of the tuple's elements into a type list
// by instantiating the type list with those types
template<class Tuple, template<class...> class TypeList>
struct unpack_tuple_into_type_list
{
  template<std::size_t... I>
  static constexpr TypeList<typename std::tuple_element<I,Tuple>::type...> impl(index_sequence<I...>);

  using type = decltype(impl(make_index_sequence<std::tuple_size<Tuple>::value>{}));
};

template<class Tuple, template<class...> class TypeList>
using unpack_tuple_into_type_list_t = typename unpack_tuple_into_type_list<Tuple,TypeList>::type;


// this metafunction takes
// 1. a list of tuples (given as a type_list of them),
// 2. an "outer" type list template, and
// 3. an "inner" type list template
// and unpacks each tuple into the inner type list,
// and collects all of those types by instantiating the outer type list
template<class ListOfTuples, template<class...> class OuterTypeList, template<class...> class InnerTypeList>
struct unpack_tuples_into_type_lists;

template<class... Tuples, template<class...> class OuterTypeList, template<class...> class InnerTypeList>
struct unpack_tuples_into_type_lists<type_list<Tuples...>, OuterTypeList, InnerTypeList>
{
  using type = OuterTypeList<unpack_tuple_into_type_list_t<Tuples, InnerTypeList>...>;
};

template<class ListOfTuples, template<class...> class OuterTypeList, template<class...> class InnerTypeList>
using unpack_tuples_into_type_lists_t = typename unpack_tuples_into_type_lists<ListOfTuples, OuterTypeList, InnerTypeList>::type;


// this metafunction takes a list of single-element type_lists
// and unwraps each "inner" type list to produce a new type_list whose
// elements are the elements from each of the inner lists
template<class ListOfSingleElementTypeLists>
struct unwrap_single_element_type_lists;

template<class... Types>
struct unwrap_single_element_type_lists<type_list<type_list<Types>...>>
{
  using type = type_list<Types...>;
};

template<class ListOfSingleElementTypeLists>
using unwrap_single_element_type_lists_t = typename unwrap_single_element_type_lists<ListOfSingleElementTypeLists>::type;


// the purpose of unpack_sender_base is to inject sender traits
// into unpack_sender when it is possible to do so (i.e., Sender is typed)
template<class Sender, class Enable = void>
struct unpack_sender_base : public sender_base {};


// If Sender is typed, define nested traits
template<class Sender>
class unpack_sender_base<Sender, typename std::enable_if<is_typed_sender<Sender>::value>::type>
{
  private:
    // collect value types from the adaptee into type_lists
    using adaptee_value_types = typename sender_traits<Sender>::template value_types<type_list,type_list>;

    // now, adaptee_value_types should look like this:
    // adaptee_value_types = type_list<type_list<tuple0>, type_list<tuple1>, ...>
    
    // unwrap each "inner" type list
    using list_of_tuples = unwrap_single_element_type_lists_t<adaptee_value_types>;

    // now, list_of_tuples should look like this:
    // list_of_tuples = type_list<tuple0, tuple1, ...>

    // value_types needs to return
    // Variant<Tuple<tuple0_element0, tuple0_element1, ...>, Tuple<tuple1_element0, tuple1_element1, ...>, ...>
    //
    // so, value_types need to unpack each tuple in list_of_tuples into value_types' Tuple template parameter
    // and collect that list into value_types' Variant template parameter

  public:
    template<template<class...> class Tuple, template<class...> class Variant>
    using value_types = unpack_tuples_into_type_lists_t<list_of_tuples, Variant, Tuple>;

    template<template<class...> class Variant>
    using error_types = typename sender_traits<Sender>::template error_types<Variant>;

    constexpr static bool sends_done = sender_traits<Sender>::sends_done;
};


template<class Sender>
class unpack_sender : public unpack_sender_base<Sender>
{
  private:
    static_assert(is_typed_sender<Sender>::value, "Sender must be typed.");

    Sender predecessor_;

  public:
    CUSEND_EXEC_CHECK_DISABLE
    template<class S,
             CUSEND_REQUIRES(std::is_constructible<Sender,S&&>::value)
            >
    CUSEND_ANNOTATION
    unpack_sender(S&& predecessor)
      : predecessor_{std::forward<S>(predecessor)}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    unpack_sender(const unpack_sender&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    unpack_sender(unpack_sender&&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    ~unpack_sender() = default;

    template<class R,
             CUSEND_REQUIRES(is_receiver<R>::value),
             CUSEND_REQUIRES(is_sender_to<Sender&&, unpack_receiver<remove_cvref_t<R>>>::value)
            >
    CUSEND_ANNOTATION
    auto connect(R&& r) &&
      -> decltype(CUSEND_NAMESPACE::connect(std::move(predecessor_), detail::make_unpack_receiver(std::forward<R>(r))))
    {
      return CUSEND_NAMESPACE::connect(std::move(predecessor_), detail::make_unpack_receiver(std::forward<R>(r)));
    }

    template<class R,
             CUSEND_REQUIRES(is_receiver<R>::value),
             CUSEND_REQUIRES(is_sender_to<const Sender&, unpack_receiver<remove_cvref_t<R>>>::value)
            >
    CUSEND_ANNOTATION
    auto connect(R&& r) const &
      -> decltype(CUSEND_NAMESPACE::connect(predecessor_, detail::make_unpack_receiver(std::forward<R>(r))))
    {
      return CUSEND_NAMESPACE::connect(predecessor_, detail::make_unpack_receiver(std::forward<R>(r)));
    }
};


template<class S,
         CUSEND_REQUIRES(is_sender_to<S, unpack_receiver<discard_receiver>>::value)
        >
CUSEND_ANNOTATION
unpack_sender<remove_cvref_t<S>>
  default_unpack(S&& predecessor)
{
  return {std::forward<S>(predecessor)};
}


template<class S>
using default_unpack_t = decltype(detail::default_unpack(std::declval<S>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"

