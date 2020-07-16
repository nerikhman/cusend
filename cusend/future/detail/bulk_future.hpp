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

#include <cstddef>
#include <type_traits>
#include <utility>
#include "../../execution/executor/executor_index.hpp"
#include "../../execution/executor/executor_shape.hpp"
#include "../../execution/executor/inline_executor.hpp"
#include "../../execution/executor/is_device_executor.hpp"
#include "../../lazy/connect.hpp"
#include "../../lazy/receiver/is_many_receiver_of.hpp"
#include "../../lazy/scheduler/schedule.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Future, class DeviceExecutor, class ManyReceiver>
class bulk_future_receiver
{
  private:
    Future future_;
    DeviceExecutor executor_;
    execution::executor_shape_t<DeviceExecutor> shape_;
    ManyReceiver receiver_;

  public:
    bulk_future_receiver(Future&& future, const DeviceExecutor& executor, execution::executor_shape_t<DeviceExecutor> shape, ManyReceiver receiver)
      : future_{std::move(future)},
        executor_{executor},
        shape_{shape},
        receiver_{receiver}
    {}

    // explicitly define this ctor to avoid viral __host__ __device__ infection of defaulted functions
    bulk_future_receiver(bulk_future_receiver&& other) noexcept
      : future_{std::move(other.future_)},
        executor_{std::move(other.executor_)},
        shape_{std::move(other.shape_)},
        receiver_{std::move(other.receiver_)}
    {}

    // explicitly define this dtor to avoid viral __host__ __device__ infection of defaulted functions
    ~bulk_future_receiver() {}

    void set_value() &&
    {
      std::move(future_).bulk_then(executor_, receiver_, shape_);
    }

    template<class E>
    void set_error(E&& e) && noexcept
    {
      CUSEND_NAMESPACE::set_error(std::move(receiver_), std::forward<E>(e));
    }

    void set_done() && noexcept
    {
      CUSEND_NAMESPACE::set_done(std::move(receiver_));
    }
};


template<class Future, class DeviceExecutor>
class bulk_future
{
  private:
    Future future_;
    DeviceExecutor executor_;
    execution::executor_shape_t<DeviceExecutor> shape_;

    using value_type = decltype(std::declval<Future>().get());

  public:
    bulk_future(Future&& future, const DeviceExecutor& executor, execution::executor_shape_t<DeviceExecutor> shape)
      : future_{std::move(future)},
        executor_{executor},
        shape_{shape}
    {}

    bulk_future(bulk_future&&) = default;


    // the only reason there are two overloads for connect() is to avoid using is_many_receiver_of with void&, which is illegal
    template<class ManyReceiver,
             class T = value_type,
             CUSEND_REQUIRES(std::is_void<T>::value),
             CUSEND_REQUIRES(is_many_receiver_of<ManyReceiver, execution::executor_index_t<DeviceExecutor>>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<ManyReceiver>::value)
            >
    auto connect(ManyReceiver receiver) &&
    {
      // wrap everything together into a bulk_future_receiver
      bulk_future_receiver<Future,DeviceExecutor,ManyReceiver> wrapped_r{std::move(future_), executor_, shape_, receiver};

      return CUSEND_NAMESPACE::connect(schedule(execution::inline_executor{}), std::move(wrapped_r));
    }


    template<class ManyReceiver,
             class T = value_type,
             CUSEND_REQUIRES(!std::is_void<T>::value),
             CUSEND_REQUIRES(is_many_receiver_of<ManyReceiver, execution::executor_index_t<DeviceExecutor>, T&>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<ManyReceiver>::value)
            >
    auto connect(ManyReceiver receiver) &&
    {
      // wrap everything together into a bulk_future_receiver
      bulk_future_receiver<Future,DeviceExecutor,ManyReceiver> wrapped_r{std::move(future_), executor_, shape_, receiver};

      return CUSEND_NAMESPACE::connect(schedule(execution::inline_executor{}), std::move(wrapped_r));
    }

    template<class OtherDeviceExecutor,
             CUSEND_REQUIRES(execution::is_device_executor<OtherDeviceExecutor>::value)
            >
    bulk_future<Future,OtherDeviceExecutor> via(const OtherDeviceExecutor& ex) &&
    {
      return {std::move(future_), ex, shape_};
    }
};


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../detail/epilogue.hpp"

