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
#include "../detail/type_traits/invoke_result.hpp"
#include "../detail/type_traits/is_invocable.hpp"
#include "../detail/type_traits/remove_cvref.hpp"
#include "../execution/executor/executor_coordinate.hpp"
#include "../execution/executor/is_device_executor.hpp"
#include "../execution/executor/kernel_executor.hpp"
#include "../lazy/detail/invocable_as_receiver.hpp"
#include "../lazy/receiver/is_receiver_of.hpp"
#include "../memory/unique_ptr.hpp"
#include "detail/bulk_future.hpp"
#include "detail/event.hpp"
#include "detail/receiver.hpp"
#include "detail/stream_of.hpp"
#include "detail/then_bulk_execute.hpp"
#include "detail/then_execute.hpp"


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
    future_base(future_base&& other) noexcept
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
    void wait()
    {
      event_.wait();
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

  protected:
    CUSEND_ANNOTATION
    future_base(const Executor& executor, bool valid, event&& e)
      : executor_{executor},
        valid_{valid},
        event_{std::move(e)}
    {}

    CUSEND_ANNOTATION
    void wait_and_invalidate()
    {
      wait();
      invalidate();
    }


    // this version of then returns our event via a move
    // the returned event corresponds to the completion
    // of set_value called on the given receiver
    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value)
            >
    CUSEND_ANNOTATION
    detail::event then_set_value_and_move_event(const DeviceExecutor& ex, R receiver) &&
    {
      // invalidate ourself
      invalidate();

      // execute the receiver on ex
      return detail::then_execute(ex, std::move(event_), receiver);
    }


    // this version of then returns a copy of our event
    // the returned event corresponds to the completion
    // of set_value called on the given receiver
    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value)
            >
    CUSEND_ANNOTATION
    detail::event then_set_value_and_copy_event(const DeviceExecutor& ex, R r) &
    {
      // execute the receiver on ex
      event_ = std::move(*this).then_set_value_and_move_event(ex, r);

      // return a new event corresponding to the completion of the execution
      return detail::event{detail::stream_of(ex)};
    }


    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(is_many_receiver_of<R, execution::executor_coordinate_t<DeviceExecutor>>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value)
            >
    CUSEND_ANNOTATION
    detail::event bulk_then_set_value_and_move_event(const DeviceExecutor& ex, R receiver, execution::executor_coordinate_t<DeviceExecutor> shape) &&
    {
      // invalidate ourself
      invalidate();

      // execute the receiver on ex
      return detail::then_bulk_execute(ex, std::move(event_), receiver, shape);
    }


  private:
    CUSEND_ANNOTATION
    void invalidate()
    {
      valid_ = false;
    }

    Executor executor_;
    bool valid_;
    detail::event event_;
};


} // end detail


// declare future for make_unready_future below
template<class T, class Executor = execution::kernel_executor>
class future;


namespace detail
{


template<class DeviceExecutor>
CUSEND_ANNOTATION
future<void, DeviceExecutor> make_unready_future(const DeviceExecutor& ex, event&& e);


template<class T, class DeviceExecutor>
CUSEND_ANNOTATION
future<T,DeviceExecutor> make_unready_future(const DeviceExecutor& ex, event&& e, memory::unique_ptr<T>&& value);


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
        // XXX it's only safe to destroy the state after event_ is complete
        //     for now, block until that happens
        //     printf to remind us we need to implement this
        // destroy_after(super_t::event_, std::move(value_));
        printf("future::~future: Blocking in destructor.\n");
        wait();
      }
    }


    future& operator=(future&&) = default;


    CUSEND_ANNOTATION
    T get() &&
    {
      super_t::wait_and_invalidate();
      return *value_;
    }


    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,T&&>::value),
             class Result = set_value_t<R,T&&>,
             CUSEND_REQUIRES(std::is_void<Result>::value)
            >
    CUSEND_ANNOTATION
    CUSEND_NAMESPACE::future<void,DeviceExecutor> then(const DeviceExecutor& ex, R receiver) &&
    {
      // close over receiver and our state
      auto wrapped_receiver = detail::make_receive_indirectly(receiver, value_.get());

      // return a future corresponding to the completion of the receiver
      // create a new event when we do this
      return detail::make_unready_future(ex, super_t::then_set_value_and_copy_event(ex, wrapped_receiver));
    }


    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,T&&>::value),
             class Result = set_value_t<R,T&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(std::is_same<Result,T>::value)
            >
    CUSEND_ANNOTATION
    future<Result,DeviceExecutor> then(const DeviceExecutor& ex, R receiver) &&
    {
      // close over receiver and our state
      auto wrapped_receiver = detail::make_receive_indirectly_inplace(receiver, value_.get());

      // return a future corresponding to the completion of the receiver
      // move the base's event when we do this
      return detail::make_unready_future(ex, std::move(*this).then_set_value_and_move_event(ex, wrapped_receiver), std::move(value_));
    }


    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,T&&>::value),
             class Result = set_value_t<R,T&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(!std::is_same<Result,T>::value)
            >
    CUSEND_ANNOTATION
    future<Result,DeviceExecutor> then(const DeviceExecutor& ex, R receiver) &&
    {
      // create storage for the result of the receiver
      // XXX this result needs to be allocated via an allocator
      cusend::memory::unique_ptr<Result> result = cusend::memory::make_unique<Result>(cusend::memory::uninitialized);

      // close over receiver and state
      auto wrapped_receiver = detail::make_receive_indirectly_and_construct_at(receiver, value_.get(), result.get());

      // return a future corresponding to the completion of the closure
      // create a new event when we do this
      return detail::make_unready_future(ex, super_t::then_set_value_and_copy_event(ex, wrapped_receiver), std::move(result));
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


    template<class DeviceExecutor,
             class Function,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<Function>::value),
             CUSEND_REQUIRES(detail::is_invocable<Function,T&&>::value),
             class Result = detail::invoke_result_t<Function,T&&>
            >
    CUSEND_ANNOTATION
    future<Result,DeviceExecutor> then(const DeviceExecutor& ex, Function f) &&
    {
      return std::move(*this).then(ex, detail::as_receiver(f));
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


    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_many_receiver_of<R, execution::executor_coordinate_t<DeviceExecutor>, T&>::value)
            >
    CUSEND_ANNOTATION
    future<T,DeviceExecutor> bulk_then(const DeviceExecutor& ex, R receiver, execution::executor_coordinate_t<DeviceExecutor> shape) &&
    {
      // close over receiver and our state
      auto wrapped_receiver = detail::make_receive_indirectly_with_coord(receiver, value_.get());

      // return a future corresponding to the completion of the receiver
      // move the base's event when we do this
      return detail::make_unready_future(ex, std::move(*this).bulk_then_set_value_and_move_event(ex, wrapped_receiver, shape), std::move(value_));
    }


    template<class R,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_many_receiver_of<R, execution::executor_coordinate_t<Executor>, T&>::value)
            >
    CUSEND_ANNOTATION
    future<T,Executor> bulk_then(R receiver, execution::executor_coordinate_t<Executor> shape) &&
    {
      return std::move(*this).bulk_then(executor(), receiver, shape);
    }


    template<class DeviceExecutor,
             class F,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<F>::value),
             CUSEND_REQUIRES(detail::is_invocable<F, execution::executor_coordinate_t<DeviceExecutor>, T&>::value)
            >
    CUSEND_ANNOTATION
    future<T,DeviceExecutor> bulk_then(const DeviceExecutor& ex, F f, execution::executor_coordinate_t<DeviceExecutor> shape) &&
    {
      return std::move(*this).bulk_then(ex, detail::as_receiver(f), shape);
    }


    template<class F,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<F>::value),
             CUSEND_REQUIRES(detail::is_invocable<F, execution::executor_coordinate_t<Executor>, T&>::value)
            >
    CUSEND_ANNOTATION
    future<T,Executor> bulk_then(F f, execution::executor_coordinate_t<Executor> shape) &&
    {
      return std::move(*this).bulk_then(executor(), f, shape);
    }


    template<class DeviceExecutor,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value)
            >
    detail::bulk_future<future, DeviceExecutor> bulk(const DeviceExecutor& ex, execution::executor_coordinate_t<DeviceExecutor> shape) &&
    {
      return {std::move(*this), ex, shape};
    }


    detail::bulk_future<future, Executor> bulk(execution::executor_coordinate_t<Executor> shape) &&
    {
      return std::move(*this).bulk(executor(), shape);
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
      super_t::wait_and_invalidate();
    }


    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             CUSEND_REQUIRES(std::is_void<set_value_t<R>>::value)
            >
    CUSEND_ANNOTATION
    future<void,DeviceExecutor> then(const DeviceExecutor& ex, R receiver) &&
    {
      // return a future corresponding to the completion of the receiver
      // move the base's event when we do this
      return detail::make_unready_future(ex, std::move(*this).then_set_value_and_move_event(ex, receiver));
    }


    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             CUSEND_REQUIRES(!std::is_void<set_value_t<R>>::value),
             class Result = set_value_t<R> 
            >
    CUSEND_ANNOTATION
    future<Result,DeviceExecutor> then(const DeviceExecutor& ex, R receiver) &&
    {
      // create storage for the result of set_value
      // XXX this result needs to be allocated via an allocator
      cusend::memory::unique_ptr<Result> result = cusend::memory::make_unique<Result>(cusend::memory::uninitialized);

      // close over receiver and the result
      auto wrapped_receiver = detail::make_receive_and_construct_at(receiver, result.get());

      // return a future corresponding to the result of the receiver
      // create a new event when we do this
      return detail::make_unready_future(ex, super_t::then_set_value_and_copy_event(ex, wrapped_receiver), std::move(result));
    }


    template<class R,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             class Result = set_value_t<R>
            >
    CUSEND_ANNOTATION
    future<Result,Executor> then(R receiver) &&
    {
      return std::move(*this).then(executor(), receiver);
    }


    template<class DeviceExecutor,
             class Function,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<Function>::value),
             CUSEND_REQUIRES(detail::is_invocable<Function>::value),
             class Result = detail::invoke_result_t<Function>
            >
    CUSEND_ANNOTATION
    future<Result,DeviceExecutor> then(const DeviceExecutor& ex, Function f) &&
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


    template<class DeviceExecutor,
             class R,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_many_receiver_of<R, execution::executor_coordinate_t<DeviceExecutor>>::value)
            >
    CUSEND_ANNOTATION
    future<void,DeviceExecutor> bulk_then(const DeviceExecutor& ex, R receiver, execution::executor_coordinate_t<DeviceExecutor> shape) &&
    {
      // return a future corresponding to the completion of the receiver
      return detail::make_unready_future(ex, std::move(*this).bulk_then_set_value_and_move_event(ex, receiver, shape));
    }


    template<class R,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_many_receiver_of<R, execution::executor_coordinate_t<Executor>>::value)
            >
    CUSEND_ANNOTATION
    future<void,Executor> bulk_then(R receiver, execution::executor_coordinate_t<Executor> shape) &&
    {
      return std::move(*this).bulk_then(executor(), receiver, shape);
    }


    template<class DeviceExecutor,
             class F,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<F>::value),
             CUSEND_REQUIRES(detail::is_invocable<F, execution::executor_coordinate_t<DeviceExecutor>>::value)
            >
    CUSEND_ANNOTATION
    future<void,DeviceExecutor> bulk_then(const DeviceExecutor& ex, F f, execution::executor_coordinate_t<DeviceExecutor> shape) &&
    {
      return std::move(*this).bulk_then(ex, detail::as_receiver(f), shape);
    }


    template<class F,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<F>::value),
             CUSEND_REQUIRES(detail::is_invocable<F, execution::executor_coordinate_t<Executor>>::value)
            >
    CUSEND_ANNOTATION
    future<void,Executor> bulk_then(F f, execution::executor_coordinate_t<Executor> shape) &&
    {
      return std::move(*this).bulk_then(executor(), f, shape);
    }


    template<class DeviceExecutor,
             CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value)
            >
    detail::bulk_future<future, DeviceExecutor> bulk(const DeviceExecutor& ex, execution::executor_coordinate_t<DeviceExecutor> shape) &&
    {
      return {std::move(*this), ex, shape};
    }


    detail::bulk_future<future, Executor> bulk(execution::executor_coordinate_t<Executor> shape) &&
    {
      return std::move(*this).bulk(executor(), shape);
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
  return future<void>{execution::kernel_executor{}, true};
}


namespace detail
{



template<class DeviceExecutor>
CUSEND_ANNOTATION
future<void,DeviceExecutor> make_unready_future(const DeviceExecutor& ex, event&& e)
{
  return {ex, true, std::move(e)};
}


template<class T, class DeviceExecutor>
CUSEND_ANNOTATION
future<T,DeviceExecutor> make_unready_future(const DeviceExecutor& ex, event&& e, memory::unique_ptr<T>&& value)
{
  return {ex, true, std::move(e), std::move(value)};
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

