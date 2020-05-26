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
#include "../detail/functional/invoke.hpp"
#include "../detail/type_traits/is_invocable.hpp"
#include "../detail/type_traits/remove_cvref.hpp"
#include "../execution/executor/stream_executor.hpp"
#include "../memory/unique_ptr.hpp"
#include "../sender/set_value.hpp"
#include "../sender/is_receiver.hpp"
#include "../sender/is_receiver_of.hpp"
#include "detail/invocable_as_receiver.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Receiver, class ValuePointer>
struct indirect_set_value
{
  Receiver r;
  ValuePointer value_ptr;

  CUSEND_ANNOTATION
  void operator()()
  {
    CUSEND_NAMESPACE::set_value(std::move(r), std::move(*value_ptr));
  }
};

template<class Receiver, class ValuePointer,
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<Receiver>::value),
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<ValuePointer>::value)
        >
CUSEND_ANNOTATION
indirect_set_value<Receiver,ValuePointer> make_indirect_set_value(Receiver r, ValuePointer value_ptr)
{
  return {r, value_ptr};
}


template<class Receiver, class ValuePointer>
struct inplace_indirect_set_value
{
  Receiver r;
  ValuePointer value_ptr;

  CUSEND_ANNOTATION
  void operator()()
  {
    *value_ptr = CUSEND_NAMESPACE::set_value(std::move(r), std::move(*value_ptr));
  }
};

template<class Receiver, class ValuePointer,
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<Receiver>::value),
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<ValuePointer>::value)
        >
CUSEND_ANNOTATION
inplace_indirect_set_value<Receiver,ValuePointer> make_inplace_indirect_set_value(Receiver r, ValuePointer value_ptr)
{
  return {r, value_ptr};
}


template<class Receiver, class ValuePointer, class ResultPointer>
struct indirect_set_value_and_construct_at
{
  Receiver r;
  ValuePointer value_ptr;
  ResultPointer result_ptr;

  CUSEND_ANNOTATION
  void operator()()
  {
    new(result_ptr) typename std::pointer_traits<ResultPointer>::element_type{CUSEND_NAMESPACE::set_value(std::move(r), std::move(*value_ptr))};
  }
};


template<class Receiver, class ValuePointer, class ResultPointer,
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<Receiver>::value),
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<ValuePointer>::value),
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<ResultPointer>::value)
        >
CUSEND_ANNOTATION
indirect_set_value_and_construct_at<Receiver,ValuePointer,ResultPointer> make_indirect_set_value_and_construct_at(Receiver r, ValuePointer value_ptr, ResultPointer result_ptr)
{
  return {r, value_ptr, result_ptr};
}


template<class Function, class ArgPointer>
struct indirect_invoke
{
  Function f;
  ArgPointer arg_ptr;

  CUSEND_ANNOTATION
  void operator()()
  {
    detail::invoke(f, std::move(*arg_ptr));
  }
};

template<class Function, class ArgPointer,
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<Function>::value),
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<ArgPointer>::value)
        >
CUSEND_ANNOTATION
indirect_invoke<Function,ArgPointer> make_indirect_invoke(Function f, ArgPointer arg_ptr)
{
  return {f, arg_ptr};
}


template<class Receiver, class ResultPointer>
struct set_value_and_construct_at
{
  Receiver r;
  ResultPointer result_ptr;

  CUSEND_ANNOTATION
  void operator()()
  {
    new(result_ptr) typename std::pointer_traits<ResultPointer>::element_type{CUSEND_NAMESPACE::set_value(std::move(r))};
  }
};

template<class Receiver, class ResultPointer,
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<Receiver>::value),
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<ResultPointer>::value)
        >
CUSEND_ANNOTATION
set_value_and_construct_at<Receiver,ResultPointer> make_set_value_and_construct_at(Receiver r, ResultPointer result_ptr)
{
  return {r, result_ptr};
}


template<class Function, class ArgPointer>
struct inplace_indirect_invoke
{
  Function f;
  ArgPointer arg_ptr;

  CUSEND_ANNOTATION
  void operator()()
  {
    *arg_ptr = detail::invoke(f, std::move(*arg_ptr));
  }
};

template<class Function, class ArgPointer,
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<Function>::value),
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<ArgPointer>::value)
        >
CUSEND_ANNOTATION
inplace_indirect_invoke<Function,ArgPointer> make_inplace_indirect_invoke(Function f, ArgPointer arg_ptr)
{
  return {f, arg_ptr};
}


template<class Function, class ArgPointer, class ResultPointer>
struct indirect_invoke_and_construct_at
{
  Function f;
  ArgPointer arg_ptr;
  ResultPointer result_ptr;

  CUSEND_ANNOTATION
  void operator()()
  {
    new(result_ptr) typename std::pointer_traits<ResultPointer>::element_type{detail::invoke(f, std::move(*arg_ptr))};
  }
};


template<class Function, class ArgPointer, class ResultPointer,
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<Function>::value),
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<ArgPointer>::value),
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<ResultPointer>::value)
        >
CUSEND_ANNOTATION
indirect_invoke_and_construct_at<Function,ArgPointer,ResultPointer> make_indirect_invoke_and_construct_at(Function f, ArgPointer arg_ptr, ResultPointer result_ptr)
{
  return {f, arg_ptr, result_ptr};
}


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


CUSEND_ANNOTATION
inline future<void> make_unready_future(const execution::stream_executor& ex, event&& e);


template<class T>
CUSEND_ANNOTATION
future<T> make_unready_future(const execution::stream_executor& ex, event&& e, memory::unique_ptr<T>&& value);


template<class Receiver>
struct call_set_value
{
  Receiver r;

  template<class... Args>
  CUSEND_ANNOTATION
  void operator()(Args&&... args)
  {
    CUSEND_NAMESPACE::set_value(std::move(r), std::forward<Args>(args)...);
  }
};


template<class Receiver,
         CUSEND_REQUIRES(is_receiver<Receiver>::value),
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<Receiver>::value)
        >
CUSEND_ANNOTATION
call_set_value<remove_cvref_t<Receiver>> make_call_set_value(Receiver r)
{
  return {r};
}


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


    template<class R,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,T&&>::value),
             class Result = set_value_t<R,T&&>,
             CUSEND_REQUIRES(std::is_void<Result>::value)
            >
    CUSEND_ANNOTATION
    CUSEND_NAMESPACE::future<void,Executor> then(R receiver) &&
    {
      // make the executor's stream wait for our event
      executor().stream_wait_for(super_t::event().native_handle());

      // close over receiver and our state
      auto closure = detail::make_indirect_set_value(receiver, value_.get());

      // execute closure on our executor
      executor().execute(closure);

      // record our event on the executor's stream
      super_t::event().record_on(executor().stream());

      // invalidate ourself
      super_t::invalidate();

      // return a future corresponding to the completion of the closure
      return detail::make_unready_future(executor(), detail::event{executor().stream()});
    }


    template<class R,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,T&&>::value),
             class Result = set_value_t<R,T&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(std::is_same<Result,T>::value)
            >
    CUSEND_ANNOTATION
    future<Result,Executor> then(R receiver) &&
    {
      // make the executor's stream wait for our event
      executor().stream_wait_for(super_t::event().native_handle());

      // close over receiver and our state
      auto closure = detail::make_inplace_indirect_set_value(receiver, value_.get());

      // execute closure on our executor
      executor().execute(closure);

      // record our event on the executor's stream
      super_t::event().record_on(executor().stream());

      // return a future corresponding to the completion of the closure
      return std::move(*this);
    }


    template<class R,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,T&&>::value),
             class Result = set_value_t<R,T&&>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(!std::is_same<Result,T>::value)
            >
    CUSEND_ANNOTATION
    future<Result,Executor> then(R receiver) &&
    {
      // make the executor's stream wait for our event
      executor().stream_wait_for(super_t::event().native_handle());

      // create storage for the result of the receiver
      // XXX needs to be allocated
      cusend::memory::unique_ptr<Result> result = cusend::memory::make_unique<Result>(cusend::memory::uninitialized);

      // close over receiver and state
      auto closure = detail::make_indirect_set_value_and_construct_at(receiver, value_.get(), result.get());

      // execute closure on our executor
      executor().execute(closure);

      // record our event on the executor's stream
      super_t::event().record_on(executor().stream());

      // invalidate ourself
      super_t::invalidate();

      // return a future corresponding to the completion of the closure
      return detail::make_unready_future(executor(), detail::event{executor().stream()}, std::move(result));
    }


    template<class Function,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<Function>::value),
             CUSEND_REQUIRES(detail::is_invocable<Function,T&&>::value),
             class Result = detail::invoke_result_t<Function,T&&>
            >
    CUSEND_ANNOTATION
    future<Result,Executor> then(Function f) &&
    {
      return std::move(*this).then(detail::as_receiver(std::forward<Function>(f)));
    }


  private:
    // give this friend access to private contructors
    template<class U>
    friend future<U> detail::make_unready_future(const execution::stream_executor& ex, detail::event&& event, memory::unique_ptr<U>&& value);

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


    template<class R,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             CUSEND_REQUIRES(std::is_same<void, set_value_t<R>>::value)
            >
    CUSEND_ANNOTATION
    future<void,Executor> then(R receiver) &&
    {
      // make the executor's stream wait for our event
      executor().stream_wait_for(super_t::event().native_handle());

      // call set_value on our executor
      executor().execute(detail::make_call_set_value(receiver));

      // record our event on the executor's stream
      super_t::event().record_on(executor().stream());

      return std::move(*this);
    }


    template<class R,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<R>::value),
             CUSEND_REQUIRES(is_receiver_of<R,void>::value),
             CUSEND_REQUIRES(!std::is_same<void, set_value_t<R>>::value),
             class Result = set_value_t<R> 
            >
    CUSEND_ANNOTATION
    future<Result,Executor> then(R receiver) &&
    {
      // make the executor's stream wait for our event
      executor().stream_wait_for(super_t::event().native_handle());

      // create storage for the result of set_value
      // XXX needs to be allocated
      cusend::memory::unique_ptr<Result> result = cusend::memory::make_unique<Result>(cusend::memory::uninitialized);

      // close over receiver and the result
      auto closure = detail::make_set_value_and_construct_at(receiver, result.get());

      // execute the closure on our executor
      executor().execute(closure);

      // record our event on the executor's stream
      super_t::event().record_on(executor().stream());

      // invalidate ourself
      super_t::invalidate();

      // return a future corresponding to the result of f
      return detail::make_unready_future(executor(), detail::event{executor().stream()}, std::move(result));
    }


    template<class Function,
             CUSEND_REQUIRES(std::is_trivially_copy_constructible<Function>::value),
             CUSEND_REQUIRES(detail::is_invocable<Function>::value),
             class Result = detail::invoke_result_t<Function>
            >
    CUSEND_ANNOTATION
    future<Result,Executor> then(Function f) &&
    {
      return std::move(*this).then(detail::as_receiver(std::forward<Function>(f)));
    }


  private:
    // give these friends access to private contructors
    friend future<void> make_ready_future();

    friend future<void> detail::make_unready_future(const execution::stream_executor& ex, detail::event&& event);

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


CUSEND_ANNOTATION
inline future<void> make_unready_future(const execution::stream_executor& ex, event&& e)
{
  return {ex, true, std::move(e)};
}


template<class T>
CUSEND_ANNOTATION
future<T> make_unready_future(const execution::stream_executor& ex, event&& e, memory::unique_ptr<T>&& value)
{
  return {ex, true, std::move(e), std::move(value)};
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

