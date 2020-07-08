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
#include <type_traits>
#include <utility>
#include "../../../detail/is_stream_executor.hpp"
#include "../../../detail/type_traits/remove_cvref.hpp"
#include "../../../future/host_promise.hpp"
#include "../../pack.hpp"
#include "../../sender/is_typed_sender.hpp"
#include "../../sender/sender_traits.hpp"
#include "../../start.hpp"
#include "../get_executor.hpp"
#include "../is_device_scheduler.hpp"
#include "detail/unpack_second_receiver.hpp"
#include "detail/variant.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class TypedSender, class DeviceScheduler>
class bulk_schedule_on_device_sender
{
  private:
    static_assert(is_typed_sender<TypedSender>::value, "TypedSender must be a typed sender.");
    static_assert(is_device_scheduler<DeviceScheduler>::value, "DeviceScheduler must be a device scheduler.");

    // XXX generalize this to executor_shape_t
    using index_type = std::size_t;

    // XXX generalize this to executor_index_t
    using shape_type = std::size_t;

    TypedSender prologue_;
    DeviceScheduler scheduler_;
    shape_type shape_;

    template<template<class...> class Tuple, template<class...> class Variant>
    using predecessor_value_types = typename sender_traits<TypedSender>::template value_types<Tuple, Variant>;

    template<template<class...> class Tuple, template<class...> class Variant>
    struct value_types_impl
    {
      // this template is passed to sender_traits::value_types below
      template<class... Types>
      using index_and_values_tuple = Tuple<index_type, Types...>;

      // the value_types this sender sends is the predecessor sender's value_types,
      // with the executor's index_type prepended into each tuple of value types
      using type = predecessor_value_types<index_and_values_tuple, Variant>;
    };


  public:
    // this sender prepends the executor index into each set of values sent from the predecessor sender
    template<template<class...> class Tuple, template<class...> class Variant>
    using value_types = typename value_types_impl<Tuple,Variant>::type;

    template<template<class...> class Variant>
    using error_types = typename sender_traits<TypedSender>::template error_types<Variant>;

    constexpr static bool sends_done = sender_traits<TypedSender>::sends_done;


    template<class OtherSender>
    bulk_schedule_on_device_sender(OtherSender&& prologue, const DeviceScheduler& scheduler, std::size_t shape)
      : prologue_{std::forward<OtherSender>(prologue)},
        scheduler_{scheduler},
        shape_{shape}
    {}


  private:
    static_assert(variant_size<value_types<tuple, variant>>::value == 1, "Predecessor sender must send exactly one tuple of values.");

    // XXX figure out what to do about the other variant alternatives
    using predecessor_value_type = variant_alternative_t<0, predecessor_value_types<tuple, variant>>;

    // the operation type returned by connect() is a composite of two separate operations
    template<class Operation1, class Operation2>
    class operation
    {
      public:
        CUSEND_EXEC_CHECK_DISABLE
        CUSEND_ANNOTATION
        operation(Operation1&& op1, Operation2&& op2)
          : op1_{std::move(op1)},
            op2_{std::move(op2)}
        {}

        // start() just start()s the two operations
        void start() &
        {
          CUSEND_NAMESPACE::start(op1_);
          CUSEND_NAMESPACE::start(op2_);
        }

      private:
        Operation1 op1_;
        Operation2 op2_;
    };

    template<class Operation1, class Operation2>
    static operation<remove_cvref_t<Operation1>, remove_cvref_t<Operation2>> make_operation(Operation1&& op1, Operation2&& op2)
    {
      return {std::forward<Operation1>(op1), std::forward<Operation2>(op2)};
    }


  public:
    // XXX this needs to constrain ManyReceiver
    template<class ManyReceiver>
    auto connect(ManyReceiver&& r) &&
    {
      host_promise<predecessor_value_type> promise;

      auto future = promise.get_future(executor());

      // create two operations
      return this->make_operation(
        // pack the prologue operation's result values into a tuple...
        CUSEND_NAMESPACE::connect(cusend::pack(std::move(prologue_)), std::move(promise)),

        // ...unpacked by the epilogue
        // note that we use an unpack_second_receiver because future.bulk() sends (index, tuple<predecessor_values...>)
        CUSEND_NAMESPACE::connect(std::move(future).bulk(shape_), detail::make_unpack_second_receiver(std::move(r)))
      );
    }


    template<class OtherDeviceScheduler,
             CUSEND_REQUIRES(is_device_scheduler<OtherDeviceScheduler>::value)
            >
    bulk_schedule_on_device_sender<TypedSender,OtherDeviceScheduler> via(const OtherDeviceScheduler& scheduler) &&
    {
      return {std::move(prologue_), scheduler, shape_};
    }


    template<class OtherDeviceScheduler,
             CUSEND_REQUIRES(is_device_scheduler<OtherDeviceScheduler>::value),
             CUSEND_REQUIRES(std::is_copy_constructible<TypedSender>::value)
            >
    bulk_schedule_on_device_sender<TypedSender,OtherDeviceScheduler> via(const OtherDeviceScheduler& scheduler) const &
    {
      return {prologue_, scheduler, shape_};
    }


    get_executor_t<DeviceScheduler> executor() const
    {
      return get_executor(scheduler_);
    }
};


template<class DeviceScheduler, class TypedSender,
         CUSEND_REQUIRES(is_device_scheduler<DeviceScheduler>::value),
         CUSEND_REQUIRES(is_typed_sender<TypedSender&&>::value)
        >
bulk_schedule_on_device_sender<remove_cvref_t<TypedSender>,DeviceScheduler>
  bulk_schedule_on_device(const DeviceScheduler& scheduler, std::size_t shape, TypedSender&& sender)
{
  return {std::forward<TypedSender>(sender), scheduler, shape};
}


template<class DeviceScheduler, class Shape, class TypedSender>
using bulk_schedule_on_device_t = decltype(detail::bulk_schedule_on_device(std::declval<DeviceScheduler>(), std::declval<Shape>(), std::declval<TypedSender>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../../detail/epilogue.hpp"

