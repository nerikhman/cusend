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
#include <utility>
#include "../../discard_receiver.hpp"
#include "../../execution/executor/callback_executor.hpp"
#include "../../just.hpp"
#include "../../memory/unique_ptr.hpp"
#include "../future.hpp"
#include "invocable_as_receiver.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
class host_future;


template<class T>
host_future<T> make_unready_host_future(const execution::callback_executor& ex, std::future<T>&& future);


host_future<void> make_unready_host_future(const execution::callback_executor& ex, std::future<void>&& future);


template<class T>
class host_future_base
{
  public:
    template<template<class...> class Tuple, template<class...> class Variant>
    using value_types = Variant<Tuple<T>>;

    template<template<class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    // explicitly define this ctor to avoid viral __host__ __device__ infection of defaulted functions
    host_future_base(host_future_base&& other)
      : executor_{std::move(other.executor_)},
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
    host_future_base(const execution::callback_executor& executor, std::future<T>&& future)
      : executor_{executor},
        future_(std::move(future))
    {}

    std::future<T>& get_future()
    {
      return future_;
    }

    const std::future<T>& get_future() const
    {
      return future_;
    }

    const execution::callback_executor& executor() const
    {
      return executor_;
    }

  private:
    execution::callback_executor executor_;
    std::future<T> future_;
};


template<class T>
class host_future : public host_future_base<T>
{
  private:
    using super_t = host_future_base<T>;
    using super_t::executor;
    using super_t::get_future;

  public:
    CUSEND_EXEC_CHECK_DISABLE
    host_future(host_future&&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    ~host_future() = default;

    using super_t::wait;
    using super_t::get;

    bool valid() const
    {
      return get_future().valid() and device_state_;
    }

    // this is the void-returning case of then(receiver)
    // XXX take an executor on which to call the receiver
    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value),
             class Result = set_value_t<R&&,T&&>,
             CUSEND_REQUIRES(std::is_void<Result>::value)
            >
    CUSEND_NAMESPACE::future<void> then(R&& receiver) &&
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      // XXX need to split the fut.get() from the receiver

      // wait and call the receiver on a stream callback
      executor().execute([fut = std::move(get_future()), receiver = std::move(receiver)] () mutable
      {
        try
        {
          CUSEND_NAMESPACE::set_value(std::move(receiver), fut.get());
        }
        catch(...)
        {
          CUSEND_NAMESPACE::set_error(std::move(receiver), std::current_exception());
        }
      });

      return detail::make_unready_future(execution::stream_executor{executor().stream()}, event{executor().stream()});
    }


    // this is the T-returning case of then(receiver)
    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value),
             class Result = set_value_t<R&&,T&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(std::is_same<Result,T>::value)
            >
    CUSEND_NAMESPACE::future<T> then(R&& receiver) &&
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      // XXX need to split the fut.get() from the receiver

      // wait and call the receiver on a stream callback
      executor().execute([fut = std::move(get_future()), receiver = std::move(receiver), d_ptr = device_state_.get()] () mutable
      {
        try
        {
          memory::construct_at(d_ptr, CUSEND_NAMESPACE::set_value(std::move(receiver), fut.get()));
        }
        catch(...)
        {
          CUSEND_NAMESPACE::set_error(std::move(receiver), std::current_exception());
        }
      });

      return detail::make_unready_future(execution::stream_executor{executor().stream()}, event{executor().stream()}, std::move(device_state_));
    }


    // this is the non-void-, non-T-returning case of then(receiver)
    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value),
             class Result = set_value_t<R&&,T&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(!std::is_same<Result,T>::value)
            >
    CUSEND_NAMESPACE::future<Result> then(R&& receiver) &&
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      // create storage for result
      memory::unique_ptr<Result> device_state = memory::make_unique<Result>(memory::uninitialized);

      // wait and call the receiver on a stream callback
      executor().execute([fut = std::move(get_future()), receiver = std::move(receiver), d_ptr = device_state.get()] () mutable
      {
        try
        {
          memory::construct_at(d_ptr, CUSEND_NAMESPACE::set_value(std::move(receiver), fut.get()));
        }
        catch(...)
        {
          CUSEND_NAMESPACE::set_error(std::move(receiver), std::current_exception());
        }
      });

      return detail::make_unready_future(execution::stream_executor{executor().stream()}, event{executor().stream()}, std::move(device_state));
    }


    // this is the invocable case of then()
    template<class F,
             CUSEND_REQUIRES(is_invocable<F,T&&>::value),
             class Result = invoke_result_t<F,T&&>
            >
    CUSEND_NAMESPACE::future<Result> then(F&& f) &&
    {
      return std::move(*this).then(detail::as_receiver(std::forward<F>(f)));
    }


    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R&&,T&&>::value)
            >
    auto connect(R&& r) &&
    {
      auto sender = CUSEND_NAMESPACE::just(std::move(*this)).then([r = std::move(r)](host_future&& self) mutable
      {
        std::move(self).then(std::move(r));
      });

      return CUSEND_NAMESPACE::connect(std::move(sender), discard_receiver{});
    }

  private:
    // give friends access to private constructor
    template<class U>
    friend host_future<U> make_unready_host_future(const execution::callback_executor& ex, std::future<U>&& future);

    host_future(execution::callback_executor executor, std::future<T>&& future)
      : super_t{executor, std::move(future)},
        // XXX need to pass an allocator here
        device_state_{memory::make_unique<T>(memory::uninitialized)}
    {}

    memory::unique_ptr<T> device_state_;
};


template<class T>
host_future<T> make_unready_host_future(const execution::callback_executor& ex, std::future<T>&& future)
{
  return {ex, std::move(future)};
}


template<>
class host_future<void> : public host_future_base<void>
{
  private:
    using super_t = host_future_base<void>;
    using super_t::executor;
    using super_t::get_future;

  public:
    host_future(host_future&&) = default;

    using super_t::wait;
    using super_t::get;

    bool valid() const
    {
      return get_future().valid();
    }

    // this is the void-returning case of then(receiver)
    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R&&,void>::value),
             class Result = set_value_t<R&&>,
             CUSEND_REQUIRES(std::is_void<Result>::value)
            >
    CUSEND_NAMESPACE::future<void> then(R&& receiver) &&
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      // wait and call the function on a stream callback
      executor().execute([fut = std::move(get_future()), receiver = std::move(receiver)] () mutable
      {
        try
        {
          fut.get();
          CUSEND_NAMESPACE::set_value(std::move(receiver));
        }
        catch(...)
        {
          CUSEND_NAMESPACE::set_error(std::move(receiver), std::current_exception());
        }
      });

      return detail::make_unready_future(execution::stream_executor{executor().stream()}, event{executor().stream()});
    }


    // this is the non-void-returning case of then(receiver)
    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R&&,void>::value),
             class Result = set_value_t<R&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value)
            >
    CUSEND_NAMESPACE::future<Result> then(R&& receiver) &&
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      // create storage for result
      memory::unique_ptr<Result> device_state = memory::make_unique<Result>(memory::uninitialized);

      // wait and call the function on a stream callback
      executor().execute([fut = std::move(get_future()), receiver = std::move(receiver), d_ptr = device_state.get()] () mutable
      {
        try
        {
          fut.get();
          memory::construct_at(d_ptr, CUSEND_NAMESPACE::set_value(std::move(receiver)));
        }
        catch(...)
        {
          CUSEND_NAMESPACE::set_error(std::move(receiver), std::current_exception());
        }
      });

      return detail::make_unready_future(execution::stream_executor{executor().stream()}, event{executor().stream()}, std::move(device_state));
    }


    // this is the invocable case of then()
    template<class F,
             CUSEND_REQUIRES(is_invocable<F>::value),
             class Result = invoke_result_t<F>
            >
    CUSEND_NAMESPACE::future<Result> then(F&& f) &&
    {
      return std::move(*this).then(detail::as_receiver(std::forward<F>(f)));
    }


    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R&&,void>::value)
            >
    auto connect(R&& r) &&
    {
      auto sender = CUSEND_NAMESPACE::just(std::move(*this)).then([r = std::move(r)](host_future&& self) mutable
      {
        std::move(self).then(std::move(r));
      });

      return CUSEND_NAMESPACE::connect(std::move(sender), discard_receiver{});
    }

  private:
    // give friends access to private constructor
    friend host_future<void> make_unready_host_future(const execution::callback_executor& ex, std::future<void>&& future);

    host_future(execution::callback_executor executor, std::future<void>&& future)
      : super_t{executor, std::move(future)}
    {}
};


inline host_future<void> make_unready_host_future(const execution::callback_executor& ex, std::future<void>&& future)
{
  return {ex, std::move(future)};
}


} // end detail

CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

