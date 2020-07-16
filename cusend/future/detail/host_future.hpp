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

#include <exception>
#include <future>
#include <type_traits>
#include <utility>
#include "../../execution/executor/callback_executor.hpp"
#include "../../execution/executor/executor_index.hpp"
#include "../../execution/executor/executor_shape.hpp"
#include "../../execution/executor/is_device_executor.hpp"
#include "../../lazy/combinator/just.hpp"
#include "../../lazy/combinator/transform.hpp"
#include "../../lazy/detail/invocable_as_receiver.hpp"
#include "../../lazy/receiver/discard_receiver.hpp"
#include "../../memory/unique_ptr.hpp"
#include "../future.hpp"
#include "bulk_future.hpp"
#include "then_bulk_execute.hpp"
#include "then_execute.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T, class Executor = execution::stream_executor>
class host_future;


template<class T, class DeviceExecutor>
host_future<T,DeviceExecutor> make_unready_host_future(const DeviceExecutor& ex, const execution::callback_executor& waiting_executor, std::future<T>&& future);


template<class DeviceExecutor>
host_future<void,DeviceExecutor> make_unready_host_future(const DeviceExecutor& ex, const execution::callback_executor& waiting_executor, std::future<void>&& future);


template<class T, class Executor>
class host_future_base
{
  public:
    template<template<class...> class Tuple, template<class...> class Variant>
    using value_types = Variant<Tuple<T>>;

    template<template<class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    // explicitly define this ctor to avoid viral __host__ __device__ infection of defaulted functions
    host_future_base(host_future_base&& other) noexcept
      : executor_{std::move(other.executor_)},
        waiting_executor_{std::move(other.waiting_executor_)},
        future_{std::move(other.future_)}
    {}

    // explicitly define this dtor to avoid viral __host__ __device__ infection of defaulted functions
    ~host_future_base() {}

    void wait()
    {
      future_.wait();
    }

    T get()
    {
      return future_.get();
    }

    const Executor& executor() const
    {
      return executor_;
    }

  protected:
    host_future_base(const Executor& ex, const execution::callback_executor& waiting_executor, std::future<T>&& future)
      : executor_{ex},
        waiting_executor_{waiting_executor},
        future_(std::move(future))
    {}

    bool valid() const
    {
      return future_.valid();
    }

    template<class Function>
    detail::event then_on_stream_callback(Function&& f) &&
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      return detail::then_execute(waiting_executor_, std::move(future_), std::forward<Function>(f));
    }

  private:
    Executor executor_;
    execution::callback_executor waiting_executor_;
    std::future<T> future_;
};


template<class T, class Executor>
class host_future : public host_future_base<T,Executor>
{
  private:
    using super_t = host_future_base<T,Executor>;

  public:
    using super_t::executor;
    using super_t::wait;
    using super_t::get;

    CUSEND_EXEC_CHECK_DISABLE
    host_future(host_future&&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    ~host_future() = default;

    bool valid() const
    {
      return super_t::valid() and device_state_;
    }

    // this is the void-returning case of then(ex, receiver)
    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             class Result = set_value_t<R&&,T&&>,
             CUSEND_REQUIRES(std::is_void<Result>::value)
            >
    future<void,DeviceExecutor> then(const DeviceExecutor& ex, R receiver) &&
    {
      // adapt the receiver
      auto adapted_receiver = detail::make_receive_indirectly(receiver, device_state_.get());

      // get an event corresponding to receiver's completion
      event e = std::move(*this).then_set_value(ex, adapted_receiver);

      // return a future
      return detail::make_unready_future(ex, std::move(e));
    }


    // this is the T-returning case of then(ex, receiver)
    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             class Result = set_value_t<R&&,T&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(std::is_same<Result,T>::value)
            >
    future<T,DeviceExecutor> then(const DeviceExecutor& ex, R receiver) &&
    {
      // adapt the receiver
      auto adapted_receiver = detail::make_receive_indirectly_inplace(receiver, device_state_.get());

      // get an event corresponding to receiver's completion
      event e = std::move(*this).then_set_value(ex, adapted_receiver);

      // return a future
      return detail::make_unready_future(ex, std::move(e), std::move(device_state_));
    }


    // this is the non-void-, non-T-returning case of then(ex, receiver)
    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             class Result = set_value_t<R&&,T&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(!std::is_same<Result,T>::value)
            >
    future<Result,DeviceExecutor> then(const DeviceExecutor& ex, R receiver) &&
    {
      // create storage for the receiver's result
      memory::unique_ptr<Result> result_state = memory::make_unique<Result>(memory::uninitialized);

      // adapt the receiver
      auto adapted_receiver = detail::make_receive_indirectly_and_construct_at(receiver, device_state_.get(), result_state.get());

      // get an event corresponding to receiver's completion
      event e = std::move(*this).then_set_value(ex, adapted_receiver);

      // return a future
      return detail::make_unready_future(ex, std::move(e), std::move(result_state));
    }


    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             class Result = set_value_t<R&&,T&&>
            >
    future<Result,Executor> then(R receiver) &&
    {
      return std::move(*this).then(executor(), receiver);
    }


    // this is the invocable case of then()
    template<class DeviceExecutor,
             class F,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(is_invocable<F,T&&>::value),
             class Result = invoke_result_t<F,T&&>
            >
    future<Result,DeviceExecutor> then(const DeviceExecutor& ex, F&& f) &&
    {
      return std::move(*this).then(ex, detail::as_receiver(std::forward<F>(f)));
    }

    template<class F,
             CUSEND_REQUIRES(is_invocable<F,T&&>::value),
             class Result = invoke_result_t<F,T&&>
            >
    future<Result,Executor> then(F&& f) &&
    {
      return std::move(*this).then(executor(), std::forward<F>(f));
    }


    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             CUSEND_REQUIRES(is_many_receiver_of<R,execution::executor_index_t<DeviceExecutor>,T&>::value)
            >
    future<T,DeviceExecutor> bulk_then(const DeviceExecutor& ex, R receiver, execution::executor_shape_t<DeviceExecutor> shape) &&
    {
      // after this future's result is ready, move it into device_state_
      detail::event event = asynchronously_move_result_to_device();

      // then execute the receiver on ex
      event = detail::then_bulk_execute(ex, std::move(event), detail::make_receive_indirectly_with_index(receiver, device_state_.get()), shape);

      // return a future corresponding to the event
      return detail::make_unready_future(ex, std::move(event), std::move(device_state_));
    }


    template<class R,
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             CUSEND_REQUIRES(is_many_receiver_of<R, execution::executor_index_t<Executor>, T&>::value)
            >
    future<T,Executor> bulk_then(R receiver, execution::executor_shape_t<Executor> shape) &&
    {
      return std::move(*this).bulk_then(executor(), receiver, shape);
    }


    template<class DeviceExecutor,
             class F,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<F>::value),
             CUSEND_REQUIRES(is_invocable<F, execution::executor_index_t<DeviceExecutor>, T&>::value)
            >
    future<T,DeviceExecutor> bulk_then(const DeviceExecutor& ex, F f, execution::executor_shape_t<DeviceExecutor> shape) &&
    {
      return std::move(*this).bulk_then(ex, detail::as_receiver(f), shape);
    }


    template<class F,
             CUSEND_REQUIRES(std::is_trivially_copyable<F>::value),
             CUSEND_REQUIRES(is_invocable<F, execution::executor_index_t<Executor>, T&>::value)
            >
    future<T,Executor> bulk_then(F f, execution::executor_shape_t<Executor> shape) &&
    {
      return std::move(*this).bulk_then(executor(), detail::as_receiver(f), shape);
    }


    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value)
            >
    auto connect(R receiver) &&
    {
      auto sender = transform(just(std::move(*this)), [receiver = std::move(receiver)](host_future&& self) mutable
      {
        std::move(self).then(receiver);
      });

      // XXX instead of the awkward composition above, a better idea would be to call .then()
      //     within a then_receiver or something wrapping receiver
      return CUSEND_NAMESPACE::connect(std::move(sender), discard_receiver{});
    }


    template<class DeviceExecutor,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value)
            >
    detail::bulk_future<host_future, DeviceExecutor> bulk(const DeviceExecutor& ex, execution::executor_shape_t<DeviceExecutor> shape) &&
    {
      return {std::move(*this), ex, shape};
    }


    detail::bulk_future<host_future, Executor> bulk(execution::executor_shape_t<Executor> shape) &&
    {
      return std::move(*this).bulk(executor(), shape);
    }


  private:
    // give friends access to private constructor
    template<class U, class E>
    friend host_future<U,E> make_unready_host_future(const E& ex, const execution::callback_executor& waiting_executor, std::future<U>&& future);

    host_future(const Executor& ex, execution::callback_executor waiting_executor, std::future<T>&& future)
      : super_t{ex, waiting_executor, std::move(future)},
        // XXX need to pass an allocator here
        device_state_{memory::make_unique<T>(memory::uninitialized)}
    {}


    detail::event asynchronously_move_result_to_device()
    {
      return std::move(*this).then_on_stream_callback([ptr = device_state_.get()](T&& value)
      {
        memory::construct_at(ptr, std::move(value));
      });
    }


    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value)
            >
    event then_set_value(const DeviceExecutor& ex, R receiver) &&
    {
      // asynchronously move the std::future's result to device
      detail::event event = asynchronously_move_result_to_device();

      // then, execute receiver on ex
      return detail::then_execute(ex, std::move(event), receiver);
    }

    memory::unique_ptr<T> device_state_;
};


template<class T, class DeviceExecutor>
host_future<T,DeviceExecutor> make_unready_host_future(const DeviceExecutor& ex, const execution::callback_executor& waiting_executor, std::future<T>&& future)
{
  static_assert(execution::is_device_executor<DeviceExecutor>::value, "make_unready_host_future: ex must be a device executor.");
  return {ex, waiting_executor, std::move(future)};
}


template<class Executor>
class host_future<void,Executor> : public host_future_base<void,Executor>
{
  private:
    using super_t = host_future_base<void,Executor>;

  public:
    using super_t::executor;
    using super_t::wait;
    using super_t::get;
    using super_t::valid;

    host_future(host_future&&) = default;

    // this is the void-returning case of then(receiver)
    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(is_receiver_of<R&&,void>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             class Result = set_value_t<R&&>,
             CUSEND_REQUIRES(std::is_void<Result>::value)
            >
    future<void,DeviceExecutor> then(const DeviceExecutor& ex, R receiver) &&
    {
      // execute the receiver on ex
      event e = std::move(*this).then_set_value(ex, receiver);

      // return a future
      return detail::make_unready_future(ex, std::move(e));
    }


    // this is the non-void-returning case of then(receiver)
    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(is_receiver_of<R&&,void>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             class Result = set_value_t<R&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value)
            >
    future<Result,DeviceExecutor> then(const DeviceExecutor& ex, R receiver) &&
    {
      // create storage for result
      memory::unique_ptr<Result> device_state = memory::make_unique<Result>(memory::uninitialized);

      // adapt the receiver
      auto adapted_receiver = detail::make_receive_and_construct_at(receiver, device_state.get());

      // execute the receiver on ex
      event e = std::move(*this).then_set_value(ex, adapted_receiver);

      // return a future
      return detail::make_unready_future(ex, std::move(e), std::move(device_state));
    }


    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R&&,void>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             class Result = set_value_t<R&&>
            >
    future<Result,Executor> then(R receiver) &&
    {
      return std::move(*this).then(executor(), receiver);
    }


    // this is the invocable case of then()
    template<class DeviceExecutor,
             class F,
             CUSEND_REQUIRES(is_invocable<F>::value),
             class Result = invoke_result_t<F>
            >
    future<Result,DeviceExecutor> then(const DeviceExecutor& ex, F&& f) &&
    {
      return std::move(*this).then(ex, detail::as_receiver(std::forward<F>(f)));
    }


    template<class F,
             CUSEND_REQUIRES(is_invocable<F>::value),
             class Result = invoke_result_t<F>
            >
    future<Result,Executor> then(F&& f) &&
    {
      return std::move(*this).then(executor(), std::forward<F>(f));
    }


    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             CUSEND_REQUIRES(is_many_receiver_of<R, execution::executor_index_t<DeviceExecutor>>::value)
            >
    future<void,DeviceExecutor> bulk_then(const DeviceExecutor& ex, R receiver, execution::executor_shape_t<DeviceExecutor> shape) &&
    {
      // on a stream callback, wait for the future to complete
      detail::event event = std::move(*this).then_on_stream_callback([]()
      {
        // no-op
      });

      // then execute the receiver on ex
      event = detail::then_bulk_execute(ex, std::move(event), receiver, shape);

      // return a future corresponding to the event
      return detail::make_unready_future(ex, std::move(event));
    }


    template<class R,
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             CUSEND_REQUIRES(is_many_receiver_of<R, execution::executor_index_t<Executor>>::value)
            >
    future<void,Executor> bulk_then(R receiver, execution::executor_shape_t<Executor> shape) &&
    {
      return std::move(*this).bulk_then(executor(), receiver, shape);
    }


    template<class DeviceExecutor,
             class F,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<F>::value),
             CUSEND_REQUIRES(is_invocable<F, execution::executor_index_t<DeviceExecutor>>::value)
            >
    future<void,DeviceExecutor> bulk_then(const DeviceExecutor& ex, F f, execution::executor_shape_t<DeviceExecutor> shape) &&
    {
      return std::move(*this).bulk_then(ex, detail::as_receiver(f), shape);
    }


    template<class F,
             CUSEND_REQUIRES(std::is_trivially_copyable<F>::value),
             CUSEND_REQUIRES(is_invocable<F, execution::executor_index_t<Executor>>::value)
            >
    future<void,Executor> bulk_then(F f, execution::executor_shape_t<Executor> shape) &&
    {
      return std::move(*this).bulk_then(executor(), detail::as_receiver(f), shape);
    }


    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R&&,void>::value)
            >
    auto connect(R&& r) &&
    {
      auto sender = transform(std::move(*this), [r = std::move(r)](host_future&& self) mutable
      {
        std::move(self).then(std::move(r));
      });

      return CUSEND_NAMESPACE::connect(std::move(sender), discard_receiver{});
    }


    template<class DeviceExecutor,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value)
            >
    detail::bulk_future<host_future, DeviceExecutor> bulk(const DeviceExecutor& ex, execution::executor_shape_t<DeviceExecutor> shape) &&
    {
      return {std::move(*this), ex, shape};
    }


    detail::bulk_future<host_future, Executor> bulk(execution::executor_shape_t<Executor> shape) &&
    {
      return std::move(*this).bulk(executor(), shape);
    }

  private:
    // give friends access to private constructor
    template<class E>
    friend host_future<void,E> make_unready_host_future(const E& ex, const execution::callback_executor& waiting_executor, std::future<void>&& future);

    host_future(const Executor& ex, execution::callback_executor waiting_executor, std::future<void>&& future)
      : super_t{ex, waiting_executor, std::move(future)}
    {}


    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value)
            >
    event then_set_value(const DeviceExecutor& ex, R receiver) &&
    {
      // on a stream callback, wait for the future to complete
      detail::event event = std::move(*this).then_on_stream_callback([]()
      {
        // no-op
      });

      // then execute the receiver on ex
      return detail::then_execute(ex, std::move(event), receiver);
    }
};


template<class DeviceExecutor>
inline host_future<void,DeviceExecutor> make_unready_host_future(const DeviceExecutor& ex, const execution::callback_executor& waiting_executor, std::future<void>&& future)
{
  static_assert(execution::is_device_executor<DeviceExecutor>::value, "make_unready_host_future: ex must be a device executor.");
  return {ex, waiting_executor, std::move(future)};
}


} // end detail

CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

