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
#include "../../detail/is_stream_executor.hpp"
#include "../../execution/executor/callback_executor.hpp"
#include "../../lazy/just.hpp"
#include "../../lazy/receiver/discard_receiver.hpp"
#include "../../lazy/transform.hpp"
#include "../../memory/unique_ptr.hpp"
#include "../future.hpp"
#include "invocable.hpp"
#include "invocable_as_receiver.hpp"
#include "then_execute.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T, class Executor = execution::stream_executor>
class host_future;


template<class T, class StreamExecutor>
host_future<T,StreamExecutor> make_unready_host_future(const StreamExecutor& ex, const execution::callback_executor& waiting_executor, std::future<T>&& future);


template<class StreamExecutor>
host_future<void,StreamExecutor> make_unready_host_future(const StreamExecutor& ex, const execution::callback_executor& waiting_executor, std::future<void>&& future);


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

  protected:
    host_future_base(const Executor& ex, const execution::callback_executor& waiting_executor, std::future<T>&& future)
      : executor_{ex},
        waiting_executor_{waiting_executor},
        future_(std::move(future))
    {}

    std::future<T> get_future()
    {
      return std::move(future_);
    }

    bool valid() const
    {
      return future_.valid();
    }

    const Executor& executor() const
    {
      return executor_;
    }

    const execution::callback_executor& waiting_executor() const
    {
      return waiting_executor_;
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
    using super_t::executor;
    using super_t::get_future;
    using super_t::waiting_executor;

  public:
    CUSEND_EXEC_CHECK_DISABLE
    host_future(host_future&&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    ~host_future() = default;

    using super_t::wait;
    using super_t::get;

    bool valid() const
    {
      return super_t::valid() and device_state_;
    }

    // this is the void-returning case of then(ex, receiver)
    template<class StreamExecutor,
             class R,
             CUSEND_REQUIRES(is_stream_executor<StreamExecutor>::value),
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             class Result = set_value_t<R&&,T&&>,
             CUSEND_REQUIRES(std::is_void<Result>::value)
            >
    future<void,StreamExecutor> then(const StreamExecutor& ex, R receiver) &&
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      // 1. on waiting_executor(), wait on the future and move the result into device memory
      detail::event event = detail::then_execute(waiting_executor(), get_future(), [ptr = device_state_.get()](T&& value)
      {
        memory::construct_at(ptr, std::move(value));
      });

      // 2. then on ex, execute indirect_set_value
      event = detail::then_execute(ex, std::move(event), detail::make_indirect_set_value(receiver, device_state_.get()));

      // return a future corresponding to the event
      return detail::make_unready_future(ex, std::move(event));
    }


    // this is the T-returning case of then(ex, receiver)
    template<class StreamExecutor,
             class R,
             CUSEND_REQUIRES(is_stream_executor<StreamExecutor>::value),
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             class Result = set_value_t<R&&,T&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(std::is_same<Result,T>::value)
            >
    future<T,StreamExecutor> then(const StreamExecutor& ex, R receiver) &&
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      // 1. on waiting_executor(), wait on the future and move the result into device memory
      detail::event event = detail::then_execute(waiting_executor(), get_future(), [ptr = device_state_.get()](T&& value)
      {
        memory::construct_at(ptr, std::move(value));
      });

      // 2. then on ex, call inplace_indirect_set_value
      event = detail::then_execute(ex, std::move(event), detail::make_inplace_indirect_set_value(receiver, device_state_.get()));

      // return a future corresponding to the event
      return detail::make_unready_future(ex, std::move(event), std::move(device_state_));
    }


    // this is the non-void-, non-T-returning case of then(ex, receiver)
    template<class StreamExecutor,
             class R,
             CUSEND_REQUIRES(is_stream_executor<StreamExecutor>::value),
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             class Result = set_value_t<R&&,T&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(!std::is_same<Result,T>::value)
            >
    future<Result,StreamExecutor> then(const StreamExecutor& ex, R receiver) &&
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      // create storage for result
      memory::unique_ptr<Result> result_state = memory::make_unique<Result>(memory::uninitialized);

      // 1. on waiting_executor(), wait on the future and move the result into device memory
      detail::event event = detail::then_execute(waiting_executor(), get_future(), [ptr = device_state_.get()](T&& value)
      {
        memory::construct_at(ptr, std::move(value));
      });

      // 2. then on ex, execute indirect_set_value_and_construct_at
      event = detail::then_execute(ex, std::move(event), detail::make_indirect_set_value_and_construct_at(receiver, device_state_.get(), result_state.get()));

      return detail::make_unready_future(ex, std::move(event), std::move(result_state));
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
    template<class StreamExecutor,
             class F,
             CUSEND_REQUIRES(is_invocable<F,T&&>::value),
             class Result = invoke_result_t<F,T&&>
            >
    future<Result,StreamExecutor> then(const StreamExecutor& ex, F&& f) &&
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


    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value)
            >
    auto connect(R receiver) &&
    {
      static_assert(std::is_trivially_copyable<R>::value, "Error.");

      auto sender = transform(just(std::move(*this)), [receiver = std::move(receiver)](host_future&& self) mutable
      {
        std::move(self).then(std::move(receiver));
      });

      return CUSEND_NAMESPACE::connect(std::move(sender), discard_receiver{});
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

    memory::unique_ptr<T> device_state_;
};


template<class T, class StreamExecutor>
host_future<T,StreamExecutor> make_unready_host_future(const StreamExecutor& ex, const execution::callback_executor& waiting_executor, std::future<T>&& future)
{
  static_assert(is_stream_executor<StreamExecutor>::value, "make_unready_host_future: ex must be a stream executor.");
  return {ex, waiting_executor, std::move(future)};
}


template<class Executor>
class host_future<void,Executor> : public host_future_base<void,Executor>
{
  private:
    using super_t = host_future_base<void,Executor>;
    using super_t::executor;
    using super_t::get_future;
    using super_t::waiting_executor;

  public:
    host_future(host_future&&) = default;

    using super_t::wait;
    using super_t::get;
    using super_t::valid;

    // this is the void-returning case of then(receiver)
    template<class StreamExecutor,
             class R,
             CUSEND_REQUIRES(is_stream_executor<StreamExecutor>::value),
             CUSEND_REQUIRES(is_receiver_of<R&&,void>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             class Result = set_value_t<R&&>,
             CUSEND_REQUIRES(std::is_void<Result>::value)
            >
    future<void,StreamExecutor> then(const StreamExecutor& ex, R receiver) &&
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      // 1. on waiting_executor(), wait on the future
      detail::event event = detail::then_execute(waiting_executor(), get_future(), []()
      {
        // no-op
      });

      // 2. then on ex, execute call_set_value
      event = detail::then_execute(ex, std::move(event), detail::make_call_set_value(receiver));

      return detail::make_unready_future(ex, std::move(event));
    }


    // this is the non-void-returning case of then(receiver)
    template<class StreamExecutor,
             class R,
             CUSEND_REQUIRES(is_stream_executor<StreamExecutor>::value),
             CUSEND_REQUIRES(is_receiver_of<R&&,void>::value),
             CUSEND_REQUIRES(std::is_trivially_copyable<R>::value),
             class Result = set_value_t<R&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value)
            >
    future<Result,StreamExecutor> then(const StreamExecutor& ex, R receiver) &&
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      // create storage for result
      memory::unique_ptr<Result> device_state = memory::make_unique<Result>(memory::uninitialized);

      // 1. on waiting_executor(), wait on the future
      detail::event event = detail::then_execute(waiting_executor(), get_future(), []()
      {
        // no-op
      });

      // 2. then on ex, execute set_value_and_construct_at
      event = detail::then_execute(ex, std::move(event), detail::make_set_value_and_construct_at(receiver, device_state.get()));

      return detail::make_unready_future(ex, std::move(event), std::move(device_state));
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
    template<class StreamExecutor,
             class F,
             CUSEND_REQUIRES(is_invocable<F>::value),
             class Result = invoke_result_t<F>
            >
    future<Result,StreamExecutor> then(const StreamExecutor& ex, F&& f) &&
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

  private:
    // give friends access to private constructor
    template<class E>
    friend host_future<void,E> make_unready_host_future(const E& ex, const execution::callback_executor& waiting_executor, std::future<void>&& future);

    host_future(const Executor& ex, execution::callback_executor waiting_executor, std::future<void>&& future)
      : super_t{ex, waiting_executor, std::move(future)}
    {}
};


template<class StreamExecutor>
inline host_future<void,StreamExecutor> make_unready_host_future(const StreamExecutor& ex, const execution::callback_executor& waiting_executor, std::future<void>&& future)
{
  static_assert(is_stream_executor<StreamExecutor>::value, "make_unready_host_future: ex must be a stream executor.");
  return {ex, waiting_executor, std::move(future)};
}


} // end detail

CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

