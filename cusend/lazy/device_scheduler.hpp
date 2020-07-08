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

#include <utility>
// XXX this trait ought to be named is_device_executor, and cudex ought to provide it
#include "../detail/is_stream_executor.hpp"
#include "detail/uncancelable_sender.hpp"
#include "detail/via_device_scheduler_sender.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


// XXX TODO make device_scheduler usable in __device__ code
//          it currently isn't because our specialization of via() internally uses std::future
template<class DeviceExecutor>
class device_scheduler
{
  public:
    explicit device_scheduler(const DeviceExecutor& executor)
      : executor_{executor}
    {}


    DeviceExecutor executor() const
    {
      return executor_;
    }


    detail::uncancelable_sender<DeviceExecutor> schedule() const
    {
      return detail::uncancelable_sender<DeviceExecutor>{executor()};
    }


  private:
    template<class Sender>
    friend detail::via_device_scheduler_sender<detail::remove_cvref_t<Sender>, device_scheduler>
      via(Sender&& predecessor, const device_scheduler& scheduler)
    {
      return {std::forward<Sender>(predecessor), scheduler};
    }
    

    friend bool operator==(const device_scheduler& lhs, const device_scheduler& rhs)
    {
      return lhs.executor_ == rhs.executor_;
    }


    friend bool operator!=(const device_scheduler& lhs, const device_scheduler& rhs)
    {
      return lhs.executor_ != rhs.executor_;
    }


    DeviceExecutor executor_;
};


template<class DeviceExecutor,
         CUSEND_REQUIRES(detail::is_stream_executor<DeviceExecutor>::value)
        >
device_scheduler<DeviceExecutor> make_device_scheduler(const DeviceExecutor& executor)
{
  return device_scheduler<DeviceExecutor>{executor};
}


namespace execution
{


// provide a specialization of as_scheduler for device executors in namespace execution
template<class DeviceExecutor,
         CUSEND_REQUIRES(CUSEND_NAMESPACE::detail::is_stream_executor<DeviceExecutor>::value)
        >
device_scheduler<DeviceExecutor> as_scheduler(const DeviceExecutor& executor)
{
  return CUSEND_NAMESPACE::make_device_scheduler(executor);
}


} // end namespace execution


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

