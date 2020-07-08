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
#include <utility>
#include "../../../detail/tuple.hpp"
#include "../../../detail/type_traits/conjunction.hpp"
#include "../../../detail/type_traits/is_detected.hpp"
#include "../../../detail/utility/index_sequence.hpp"
#include "../../../execution/executor/bulk_execute.hpp"
#include "../../receiver/is_many_receiver_of.hpp"
#include "../../receiver/set_done.hpp"
#include "../../receiver/set_error.hpp"
#include "../../receiver/set_value.hpp"
#include "../../sender/sender_traits.hpp"
#include "optional.hpp"
#include "variant.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{
namespace fan_out_receiver_detail
{


template<class R, class Tuple>
struct is_many_receiver_of_tupled_arguments;

template<class R, class... Args>
struct is_many_receiver_of_tupled_arguments<R, tuple<Args...>> : is_many_receiver_of<R,Args...> {};


template<class R, class VariantOfTuples>
struct is_many_receiver_of_all_of;

template<class R, class... Tuples>
struct is_many_receiver_of_all_of<R, variant<Tuples...>> : detail::conjunction<is_many_receiver_of_tupled_arguments<R, Tuples>...> {};


template<class O, class... Args>
using emplace_t = decltype(std::declval<O>.emplace(std::declval<Args>()...));

template<class O, class... Args>
using can_emplace = is_detected<emplace_t, O, Args...>;


} // end fan_out_receiver_detail


template<class TypedSingleSender, class Executor, class ManyReceiver>
class fan_out_receiver
{
  private:
    using sender_type = remove_cvref_t<TypedSingleSender>;
    using receiver_type = remove_cvref_t<ManyReceiver>;
    using variant_of_tuples_type = typename sender_traits<sender_type>::template value_types<tuple, variant>;
  
    detail::optional<variant_of_tuples_type> maybe_variant_of_tuples_;
    Executor ex_;
    std::size_t shape_;
    receiver_type receiver_;

  public:
    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    fan_out_receiver(const Executor& ex, std::size_t shape, receiver_type&& receiver)
      : maybe_variant_of_tuples_{},
        ex_{ex},
        shape_{shape},
        receiver_{std::move(receiver)}
    {}


    fan_out_receiver(fan_out_receiver&&) = default;

  
    template<class Error,
             CUSEND_REQUIRES(is_receiver<receiver_type&&, Error&&>::value)
            >
    CUSEND_ANNOTATION
    void set_error(Error&& e) &&
    {
      CUSEND_NAMESPACE::set_error(std::move(receiver_), std::forward<Error>(e));
    }

  
    CUSEND_ANNOTATION
    void set_done() &&
    {
      CUSEND_NAMESPACE::set_done(std::move(receiver_));
    }


  private:
    template<class... Args>
    struct set_value_from_tuple
    {
      detail::tuple<Args...> args;
      receiver_type receiver;

      template<std::size_t... I,
               CUSEND_REQUIRES(is_many_receiver_of<receiver_type, std::size_t, Args&...>::value)
              >
      CUSEND_ANNOTATION
      void impl(std::size_t i, index_sequence<I...>)
      {
        CUSEND_NAMESPACE::set_value(receiver, i, detail::get<I>(args)...);
      }

      template<CUSEND_REQUIRES(is_many_receiver_of<receiver_type, std::size_t, Args&...>::value)>
      CUSEND_ANNOTATION
      void operator()(std::size_t i)
      {
        this->impl(i, index_sequence_for<Args...>{});
      }
    };


    struct set_value_visitor
    {
      Executor ex;
      std::size_t shape;
      receiver_type receiver;

      template<class... Args>
      CUSEND_ANNOTATION
      void operator()(detail::tuple<Args...>& args)
      {
        set_value_from_tuple<Args...> f{std::move(args), std::move(receiver)};

        execution::bulk_execute(ex, std::move(f), shape);
      }
    };


  public:
    template<class... Args,
             CUSEND_REQUIRES(is_many_receiver_of<receiver_type, std::size_t, remove_cvref_t<Args>&...>::value)
             // XXX not sure how to spell this requirement
             //, CUSEND_REQUIRES(fan_out_receiver_detail::can_emplace<optional<variant_of_tuples_type>, tuple<Args&&...>>::value)
            >
    CUSEND_ANNOTATION
    void set_value(Args&&... args) &&
    {
      // emplace the result
      maybe_variant_of_tuples_.emplace(detail::make_tuple(std::forward<Args>(args)...));
  
      // visit the variant of tuples
      detail::visit(set_value_visitor{ex_, shape_, std::move(receiver_)}, *maybe_variant_of_tuples_);
    }
};


namespace fan_out_receiver_detail
{


template<class Index, class Tuple>
struct index_and_lvalue_references;

template<class Index, class... Types>
struct index_and_lvalue_references<Index, tuple<Types...>>
{
  using type = tuple<Index, typename std::add_lvalue_reference<Types>::type...>;
};


template<class Index, class Tuple>
using index_and_lvalue_references_t = typename index_and_lvalue_references<Index,Tuple>::type;


template<class VariantOfTuples>
struct bulk_sender_value_types;

template<class... Tuples>
struct bulk_sender_value_types<variant<Tuples...>>
{
  using type = variant<index_and_lvalue_references_t<std::size_t, Tuples>...>;
};

template<class VariantOfTuples>
using bulk_sender_value_types_t = typename bulk_sender_value_types<VariantOfTuples>::type;


} // end fan_out_receiver_detail


template<class TypedSender, class Executor, class ManyReceiver,
         // require that the sender is typed...
         CUSEND_REQUIRES(is_typed_sender<TypedSender&&>::value)

         // get the type of values sent by the sender
         , class SenderValueTypes = typename sender_traits<TypedSender&&>::template value_types<tuple, variant>

         // prepend an index to the values sent by the sender and add a lvalue reference to the elements in the tail
         , class BulkSenderValueTypes = fan_out_receiver_detail::bulk_sender_value_types_t<SenderValueTypes>

         // require that ex is an executor
         , CUSEND_REQUIRES(execution::is_executor<Executor>::value)

         // require that the receiver is many and can receive the bulk value types
         , CUSEND_REQUIRES(fan_out_receiver_detail::is_many_receiver_of_all_of<ManyReceiver&&,BulkSenderValueTypes>::value)
        >
CUSEND_ANNOTATION
fan_out_receiver<TypedSender,Executor,ManyReceiver> make_fan_out_receiver(const Executor& ex, std::size_t shape, ManyReceiver&& r)
{
  return {ex, shape, std::forward<ManyReceiver>(r)};
}


template<class Sender, class Executor, class Shape, class Receiver>
using make_fan_out_receiver_t = decltype(detail::make_fan_out_receiver<Sender>(std::declval<Executor>(), std::declval<Shape>(), std::declval<Receiver>()));


template<class Sender, class Executor, class Shape, class Receiver>
using can_make_fan_out_receiver = is_detected<make_fan_out_receiver_t, Sender&&, Executor, Shape, Receiver>;


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"

