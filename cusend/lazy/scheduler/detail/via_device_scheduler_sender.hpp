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

#include <type_traits>
#include "../../../detail/type_traits/remove_cvref.hpp"
#include "../../../future/future.hpp"
#include "../../../future/host_promise.hpp"
#include "../../pack.hpp"
#include "../../receiver/is_receiver_of.hpp"
#include "../../sender/sender_traits.hpp"
#include "../../start.hpp"
#include "../../unpack.hpp"
#include "../get_executor.hpp"
#include "../is_device_scheduler.hpp"
#include "detail/variant.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class TypedSender, class DeviceScheduler>
class via_device_scheduler_sender
{
  private:
    static_assert(is_typed_sender<TypedSender>::value, "TypedSender must be a typed sender.");
    static_assert(is_device_scheduler<DeviceScheduler>::value, "DeviceScheduler must be a device scheduler.");

    TypedSender prologue_;
    DeviceScheduler scheduler_;

  public:
    template<template<class...> class Tuple, template<class...> class Variant>
    using value_types = typename sender_traits<TypedSender>::template value_types<Tuple,Variant>;

    template<template<class...> class Variant>
    using error_types = typename sender_traits<TypedSender>::template error_types<Variant>;

    constexpr static bool sends_done = sender_traits<TypedSender>::sends_done;

    template<class OtherSender>
    via_device_scheduler_sender(OtherSender&& prologue, const DeviceScheduler& scheduler)
      : prologue_{std::move(prologue)},
        scheduler_{scheduler}
    {}

  private:
    static_assert(variant_size<value_types<tuple, variant>>::value == 1, "Predecessor sender must send exactly one type of value.");

    // XXX figure out what to do about the other variant alternatives
    using value_type = variant_alternative_t<0, value_types<tuple, variant>>;

    // the operation type returned by connect() is a composite of two separate operations
    template<class Operation1, class Operation2>
    class operation
    {
      public:
        operation(Operation1&& op1, Operation2&& op2)
          : op1_{std::move(op1)},
            op2_{std::move(op2)}
        {}

        // start() just start()s the two operations
        void start()
        {
          cusend::start(op1_);
          cusend::start(op2_);
        }

      private:
        Operation1 op1_;
        Operation2 op2_;
    };

    using executor_type = get_executor_t<DeviceScheduler>;

    // these two aliases are shorthands for the operations composed by connect()
    // the purpose of naming them is so that we can name the type of result of connect()
    using operation1_type = connect_t<pack_t<TypedSender>, host_promise<value_type>>;

    template<class Receiver>
    using operation2_type = connect_t<unpack_t<typename host_promise<value_type>::template future_type<executor_type>>, Receiver>;


  public:
    template<class Receiver,
             CUSEND_REQUIRES(is_sender_to<TypedSender,Receiver>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<Receiver>::value)
            >
    operation<operation1_type, operation2_type<Receiver>> connect(Receiver r) &&
    {
      host_promise<value_type> promise;

      auto future = promise.get_future(executor());

      // create two operations
      return {
        // pack the prologue operation's result values into a tuple...
        cusend::connect(pack(std::move(prologue_)), std::move(promise)),

        // ...unpacked by the epilogue
        cusend::connect(unpack(std::move(future)), std::move(r))
      };
    }

    template<class OtherDeviceScheduler,
             CUSEND_REQUIRES(is_device_scheduler<OtherDeviceScheduler>::value)
            >
    via_device_scheduler_sender<TypedSender,OtherDeviceScheduler> via(const OtherDeviceScheduler& scheduler) &&
    {
      return {std::move(prologue_), scheduler};
    }

    template<class OtherDeviceScheduler,
             CUSEND_REQUIRES(is_device_scheduler<OtherDeviceScheduler>::value),
             CUSEND_REQUIRES(std::is_copy_constructible<TypedSender>::value)
            >
    via_device_scheduler_sender<TypedSender,OtherDeviceScheduler> via(const OtherDeviceScheduler& scheduler) const &
    {
      return {prologue_, scheduler};
    }

    get_executor_t<DeviceScheduler> executor() const
    {
      return get_executor(scheduler_);
    }
};


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../../detail/epilogue.hpp"

