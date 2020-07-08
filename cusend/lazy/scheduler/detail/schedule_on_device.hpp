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

#include <exception>
#include <type_traits>
#include <utility>
#include "../../../detail/is_stream_executor.hpp"
#include "../../../execution/executor/execute.hpp"
#include "../../detail/receiver_as_trivially_copyable_invocable.hpp"
#include "../get_executor.hpp"
#include "../is_device_scheduler.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


// this sender never calls set_done() (in a destructor, for example)
// thus, the invocable executed inside of start() is trivially copyable
// trivially copyable invocables are required by device executors
template<class DeviceScheduler>
class schedule_on_device_sender
{
  private:
    static_assert(is_device_scheduler<DeviceScheduler>::value, "DeviceScheduler must be a device scheduler.");

    DeviceScheduler scheduler_;

    using executor_type = get_executor_t<DeviceScheduler>;

  private:
    template<class R>
    class operation
    {
      private:
        executor_type ex_;
        R receiver_;

      public:
        CUSEND_EXEC_CHECK_DISABLE
        CUSEND_ANNOTATION
        operation(const executor_type& ex, R receiver)
          : ex_{ex},
            receiver_{receiver}
        {}

#if CUSEND_HAS_EXCEPTIONS
        CUSEND_ANNOTATION
        void start() & noexcept
        {
          try
          {
            execution::execute(ex_, detail::as_trivially_copyable_invocable(receiver_));
          }
          catch(...)
          {
            set_error(std::move(receiver_), std::current_exception());
          }
        }
#else
        CUSEND_ANNOTATION
        void start() & noexcept
        {
          execution::execute(ex_, detail::as_trivially_copyable_invocable(receiver_));
        }
#endif
    };

  public:
    template<template<class...> class Tuple, template<class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template<template<class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    constexpr static bool sends_done = false;


    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    schedule_on_device_sender(const DeviceScheduler& scheduler)
      : scheduler_{scheduler}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    schedule_on_device_sender(const schedule_on_device_sender&) = default;


    CUSEND_ANNOTATION
    executor_type executor() const
    {
      return get_executor(scheduler_);
    }


    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value)
            >
    CUSEND_ANNOTATION
    operation<R> connect(R receiver) const
    {
      return {executor(), std::forward<R>(receiver)};
    }


    template<class OtherDeviceScheduler,
             CUSEND_REQUIRES(is_device_scheduler<OtherDeviceScheduler>::value)
            >
    schedule_on_device_sender<OtherDeviceScheduler> via(const OtherDeviceScheduler& scheduler) const
    {
      return {scheduler};
    }


    template<class DeviceExecutor,
             CUSEND_REQUIRES(is_stream_executor<DeviceExecutor>::value)
            >
    schedule_on_device_sender<as_scheduler_t<DeviceExecutor>>
      on(const DeviceExecutor& executor) const
    {
      return {as_scheduler(executor)};
    }
};


// this function does not constrain DeviceScheduler with is_device_scheduler
// because that would create a circular dependency between is_device_scheduler and device_scheduler::schedule
template<class DeviceScheduler>
schedule_on_device_sender<DeviceScheduler>
  schedule_on_device(const DeviceScheduler& scheduler)
{
  return {scheduler};
}


template<class DeviceScheduler>
using schedule_on_device_t = decltype(detail::schedule_on_device(std::declval<DeviceScheduler>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../../detail/epilogue.hpp"

