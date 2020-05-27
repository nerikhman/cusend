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

#include <memory>
#include <type_traits>
#include <utility>
#include "../detail/event.hpp"
#include "../detail/type_traits/invoke_result.hpp"
#include "../detail/type_traits/is_invocable.hpp"
#include "../detail/type_traits/remove_cvref.hpp"
#include "../execution/executor/stream_executor.hpp"
#include "../memory/unique_ptr.hpp"
#include "../sender/set_value.hpp"
#include "../sender/is_receiver.hpp"
#include "../sender/is_receiver_of.hpp"
#include "detail/invocable.hpp"
#include "detail/invocable_as_receiver.hpp"
#include "detail/is_stream_executor.hpp"
#include "detail/stream_of.hpp"
#include "detail/stream_wait_for.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Executor>
class future_base
{
  public:
    CUSEND_ANNOTATION
    explicit future_base(const Executor& executor)
      : future_base{executor, false, detail::event{}}
    {}

    CUSEND_ANNOTATION
    future_base()
      : future_base(Executor{})
    {}

    CUSEND_ANNOTATION
    future_base(future_base&& other)
      : executor_{std::move(other.executor_)},
        valid_{other.valid_},
        event_{std::move(other.event_)}
    {
      other.valid_ = false;
    }

    CUSEND_ANNOTATION
    future_base& operator=(future_base&& other)
    {
      executor_ = std::move(other.executor_);
      valid_ = other.valid_;
      event_ = std::move(other.event_);
      other.valid_ = false;
      return *this;
    }

    CUSEND_ANNOTATION
    const Executor& executor() const
    {
      return executor_;
    }

    CUSEND_ANNOTATION
    bool valid() const
    {
      return valid_;
    }

    CUSEND_ANNOTATION
    bool is_ready() const
    {
      return valid() and event_.is_ready();
    }

    CUSEND_ANNOTATION
    void wait()
    {
      event_.wait();
    }

  protected:
    CUSEND_ANNOTATION
    future_base(const Executor& executor, bool valid, event&& e)
      : executor_{executor},
        valid_{valid},
        event_{std::move(e)}
    {}

    CUSEND_ANNOTATION
    void invalidate()
    {
      valid_ = false;
    }

    CUSEND_ANNOTATION
    detail::event& event()
    {
      return event_;
    }

  private:
    Executor executor_;
    bool valid_;
    detail::event event_;
};


} // end detail


// declare future for make_unready_future below
template<class T, class Executor = execution::stream_executor>
class future;


namespace detail
{


template<class StreamExecutor>
CUSEND_ANNOTATION
future<void, StreamExecutor> make_unready_future(const StreamExecutor& ex, event&& e);


template<class T, class StreamExecutor>
CUSEND_ANNOTATION
future<T,StreamExecutor> make_unready_future(const StreamExecutor& ex, event&& e, memory::unique_ptr<T>&& value);


} // end detail


template<class T, class Executor>
class future : private detail::future_base<Executor>
{
  private:
    using super_t = detail::future_base<Executor>;

  public:
    using super_t::executor;
    using super_t::valid;
    using super_t::is_ready;
    using super_t::wait;

    future() = default;


    future(future&&) = default;


    CUSEND_ANNOTATION
    explicit future(T&& value)
      : super_t{Executor{}, true, detail::event{}},
        value_{cusend::memory::make_unique<T>(std::move(value))}
    {}


    CUSEND_ANNOTATION
    ~future()
    {
      if(value_.get())
      {
        // destroy_after(super_t::event(), std::move(value_));
        printf("future::~future: Blocking in destructor.\n");
        super_t::event().wait();
      }
    }


    future& operator=(future&&) = default;


    CUSEND_ANNOTATION
    T get() &&
    {
      wait();
      super_t::invalidate();
      return *value_;
    }


    template<class StreamExecutor,
             class R,
             CUSEND_REQUIRES(detail::is_stream_executor<StreamExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,T&&>::value),
             class Result = set_value_t<R,T&&>,
             CUSEND_REQUIRES(std::is_void<Result>::value)
            >
    CUSEND_ANNOTATION
    CUSEND_NAMESPACE::future<void,StreamExecutor> then(const StreamExecutor& ex, R receiver) &&
    {
      // get the executor's stream
      cudaStream_t stream = detail::stream_of(ex);

      // make the stream wait for our event
      detail::stream_wait_for(stream, super_t::event().native_handle());

      // close over receiver and our state
      auto closure = detail::make_indirect_set_value(receiver, value_.get());

      // execute closure on the executor
      execution::execute(ex, closure);

      // record our event on the stream
      super_t::event().record_on(stream);

      // invalidate ourself
      super_t::invalidate();

      // return a future corresponding to the completion of the closure
      return detail::make_unready_future(ex, detail::event{stream});
    }


    template<class StreamExecutor,
             class R,
             CUSEND_REQUIRES(detail::is_stream_executor<StreamExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,T&&>::value),
             class Result = set_value_t<R,T&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(std::is_same<Result,T>::value)
            >
    CUSEND_ANNOTATION
    future<Result,StreamExecutor> then(const StreamExecutor& ex, R receiver) &&
    {
      // get the executor's stream
      cudaStream_t stream = detail::stream_of(ex);

      // make the stream wait for our event
      detail::stream_wait_for(stream, super_t::event().native_handle());

      // close over receiver and our state
      auto closure = detail::make_inplace_indirect_set_value(receiver, value_.get());

      // execute closure on our executor
      execution::execute(ex, closure);

      // record our event on the stream
      super_t::event().record_on(stream);

      // invalidate ourself
      super_t::invalidate();

      // return a future corresponding to the completion of the closure
      return detail::make_unready_future(ex, std::move(super_t::event()), std::move(value_));
    }


    template<class StreamExecutor,
             class R,
             CUSEND_REQUIRES(detail::is_stream_executor<StreamExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,T&&>::value),
             class Result = set_value_t<R,T&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(!std::is_same<Result,T>::value)
            >
    CUSEND_ANNOTATION
    future<Result,StreamExecutor> then(const StreamExecutor& ex, R receiver) &&
    {
      // get the executor's stream
      cudaStream_t stream = detail::stream_of(ex);

      // make the executor's stream wait for our event
      detail::stream_wait_for(stream, super_t::event().native_handle());

      // create storage for the result of the receiver
      // XXX needs to be allocated
      cusend::memory::unique_ptr<Result> result = cusend::memory::make_unique<Result>(cusend::memory::uninitialized);

      // close over receiver and state
      auto closure = detail::make_indirect_set_value_and_construct_at(receiver, value_.get(), result.get());

      // execute closure on our executor
      execution::execute(ex, closure);

      // record our event on the stream
      super_t::event().record_on(stream);

      // invalidate ourself
      super_t::invalidate();

      // return a future corresponding to the completion of the closure
      return detail::make_unready_future(ex, detail::event{executor().stream()}, std::move(result));
    }


    template<class R,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,T&&>::value),
             class Result = set_value_t<R,T&&>
            >
    CUSEND_ANNOTATION
    future<Result,Executor> then(R receiver) &&
    {
      return std::move(*this).then(executor(), receiver);
    }


    template<class StreamExecutor,
             class Function,
             CUSEND_REQUIRES(detail::is_stream_executor<StreamExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<Function>::value),
             CUSEND_REQUIRES(detail::is_invocable<Function,T&&>::value),
             class Result = detail::invoke_result_t<Function,T&&>
            >
    CUSEND_ANNOTATION
    future<Result,StreamExecutor> then(const StreamExecutor& ex, Function f) &&
    {
      return std::move(*this).then(ex, detail::as_receiver(std::forward<Function>(f)));
    }


    template<class Function,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<Function>::value),
             CUSEND_REQUIRES(detail::is_invocable<Function,T&&>::value),
             class Result = detail::invoke_result_t<Function,T&&>
            >
    CUSEND_ANNOTATION
    future<Result,Executor> then(Function f) &&
    {
      return std::move(*this).then(executor(), f);
    }


  private:
    // give this friend access to private contructors
    template<class U, class E>
    friend future<U,E> detail::make_unready_future(const E& ex, detail::event&& event, memory::unique_ptr<U>&& value);

    CUSEND_ANNOTATION
    future(const Executor& executor, bool valid, detail::event&& event, cusend::memory::unique_ptr<T>&& value)
      : super_t{executor, valid, std::move(event)},
        value_{std::move(value)}
    {}

    // XXX this needs to have an allocation_deleter
    cusend::memory::unique_ptr<T> value_;
};


template<class T>
CUSEND_ANNOTATION
future<detail::remove_cvref_t<T>> make_ready_future(T&& value)
{
  return future<detail::remove_cvref_t<T>>{std::forward<T>(value)};
}


CUSEND_ANNOTATION
future<void> make_ready_future();


template<class Executor>
class future<void,Executor> : private detail::future_base<Executor>
{
  private:
    using super_t = detail::future_base<Executor>;

  public:
    using super_t::executor;
    using super_t::valid;
    using super_t::is_ready;
    using super_t::wait;

    future() = default;


    CUSEND_ANNOTATION
    void get() &&
    {
      wait();
      super_t::invalidate();
    }


    template<class StreamExecutor,
             class R,
             CUSEND_REQUIRES(detail::is_stream_executor<StreamExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             CUSEND_REQUIRES(std::is_same<void, set_value_t<R>>::value)
            >
    CUSEND_ANNOTATION
    future<void,StreamExecutor> then(const StreamExecutor& ex, R receiver) &&
    {
      // get the executor's stream
      cudaStream_t stream = detail::stream_of(ex);

      // make the stream wait for our event
      detail::stream_wait_for(stream, super_t::event().native_handle());

      // call set_value on the executor
      execution::execute(ex, detail::make_call_set_value(receiver));

      // record our event on the stream
      super_t::event().record_on(stream);

      // invalidate ourself
      super_t::invalidate();

      // return a future corresponding to the completion of the receiver
      return detail::make_unready_future(ex, std::move(super_t::event()));
    }


    template<class StreamExecutor,
             class R,
             CUSEND_REQUIRES(detail::is_stream_executor<StreamExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             CUSEND_REQUIRES(!std::is_same<void, set_value_t<R>>::value),
             class Result = set_value_t<R> 
            >
    CUSEND_ANNOTATION
    future<Result,StreamExecutor> then(const StreamExecutor& ex, R receiver) &&
    {
      // get the executor's stream
      cudaStream_t stream = detail::stream_of(ex);

      // make the stream wait for our event
      detail::stream_wait_for(stream, super_t::event().native_handle());

      // create storage for the result of set_value
      // XXX needs to be allocated
      cusend::memory::unique_ptr<Result> result = cusend::memory::make_unique<Result>(cusend::memory::uninitialized);

      // close over receiver and the result
      auto closure = detail::make_set_value_and_construct_at(receiver, result.get());

      // execute the closure on the executor
      execution::execute(ex, closure);

      // record our event on the stream
      super_t::event().record_on(stream);

      // invalidate ourself
      super_t::invalidate();

      // return a future corresponding to the result of f
      return detail::make_unready_future(ex, detail::event{stream}, std::move(result));
    }


    template<class R,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             class Result = set_value_t<R,void>
            >
    CUSEND_ANNOTATION
    future<Result,Executor> then(R receiver) &&
    {
      return std::move(*this).then(executor(), receiver);
    }


    template<class StreamExecutor,
             class Function,
             CUSEND_REQUIRES(detail::is_stream_executor<StreamExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<Function>::value),
             CUSEND_REQUIRES(detail::is_invocable<Function>::value),
             class Result = detail::invoke_result_t<Function>
            >
    CUSEND_ANNOTATION
    future<Result,StreamExecutor> then(const StreamExecutor& ex, Function f) &&
    {
      return std::move(*this).then(ex, detail::as_receiver(std::forward<Function>(f)));
    }


    template<class Function,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<Function>::value),
             CUSEND_REQUIRES(detail::is_invocable<Function>::value),
             class Result = detail::invoke_result_t<Function>
            >
    CUSEND_ANNOTATION
    future<Result,Executor> then(Function f) &&
    {
      return std::move(*this).then(executor(), f);
    }


  private:
    // give these friends access to private contructors
    friend future<void> make_ready_future();

    template<class E>
    friend future<void,E> detail::make_unready_future(const E& ex, detail::event&& event);

    CUSEND_ANNOTATION
    future(const Executor& executor, bool valid, detail::event&& event)
      : super_t{executor, valid, std::move(event)}
    {}

    CUSEND_ANNOTATION
    future(const Executor& executor, bool valid)
      : future{executor, valid, detail::event{}}
    {}
};


CUSEND_ANNOTATION
inline future<void> make_ready_future()
{
  return future<void>{execution::stream_executor{}, true};
}


namespace detail
{



template<class StreamExecutor>
CUSEND_ANNOTATION
future<void,StreamExecutor> make_unready_future(const StreamExecutor& ex, event&& e)
{
  return {ex, true, std::move(e)};
}


template<class T, class StreamExecutor>
CUSEND_ANNOTATION
future<T,StreamExecutor> make_unready_future(const StreamExecutor& ex, event&& e, memory::unique_ptr<T>&& value)
{
  return {ex, true, std::move(e), std::move(value)};
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

