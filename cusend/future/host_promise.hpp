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
#include "../execution/executor/callback_executor.hpp"
#include "detail/host_future.hpp"
#include "detail/is_stream_executor.hpp"
#include "detail/stream_of.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


template<class T>
class host_promise
{
  public:
    CUSEND_EXEC_CHECK_DISABLE
    host_promise() = default;

    CUSEND_EXEC_CHECK_DISABLE
    host_promise(host_promise&&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    ~host_promise() = default;

    template<class U = T,
             CUSEND_REQUIRES(std::is_void<U>::value)
            >
    void set_value() &&
    {
      promise_.set_value();
    }

    template<class U = T,
             CUSEND_REQUIRES(!std::is_void<U>::value)
            >
    void set_value(const U& value) &&
    {
      promise_.set_value(value);
    }

    template<class U = T,
             CUSEND_REQUIRES(!std::is_void<U>::value)
            >
    void set_value(U&& value) &&
    {
      promise_.set_value(std::move(value));
    }

    void set_error(std::exception_ptr e) && noexcept
    {
      promise_.set_exception(std::move(e));
    }

    void set_done() && noexcept 
    {
      // indicate that we've abandoned the shared state
      std::future_error error{std::future_errc::broken_promise};

      // signal via the error channel
      std::move(*this).set_error(std::make_exception_ptr(error));
    }

    template<class StreamExecutor,
             CUSEND_REQUIRES(detail::is_stream_executor<StreamExecutor>::value)
            >
    detail::host_future<T,StreamExecutor> get_future(const StreamExecutor& executor, const execution::callback_executor& waiting_executor)
    {
      return detail::make_unready_host_future(executor, waiting_executor, promise_.get_future());
    }

    template<class StreamExecutor,
             CUSEND_REQUIRES(detail::is_stream_executor<StreamExecutor>::value)
            >
    detail::host_future<T,StreamExecutor> get_future(const StreamExecutor& executor)
    {
      return this->get_future(executor, execution::callback_executor{detail::stream_of(executor)});
    }

    detail::host_future<T,execution::stream_executor> get_future()
    {
      return this->get_future(execution::stream_executor{});
    }

  private:
    std::promise<T> promise_;
};


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

