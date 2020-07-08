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

#include "../../../../detail/prologue.hpp"

#include <exception>
#include <utility>
#include "../../../../detail/type_traits/remove_cvref.hpp"
#include "../../../../execution/executor/is_executor.hpp"
#include "../../../receiver/is_receiver_of.hpp"
#include "execute_operation.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Executor>
class executor_as_sender
{
  public:
    template<template<class...> class Tuple, template<class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template<template<class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    constexpr static bool sends_done = true;


    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    executor_as_sender(const Executor& executor)
      : executor_{executor}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    executor_as_sender(const executor_as_sender&) = default;


    CUSEND_ANNOTATION
    const Executor& executor() const
    {
      return executor_;
    }


    template<class R,
             CUSEND_REQUIRES(is_receiver_of<R,void>::value)
            >
    CUSEND_ANNOTATION
    execute_operation<Executor,remove_cvref_t<R&&>> connect(R&& receiver) const
    {
      return detail::make_execute_operation(executor_, std::forward<R>(receiver));
    }


    template<class OtherExecutor,
             CUSEND_REQUIRES(execution::is_executor<OtherExecutor>::value)
            >
    CUSEND_ANNOTATION
    executor_as_sender<OtherExecutor> on(const OtherExecutor& ex) const
    {
      return {ex};
    }


  private:
    Executor executor_;
};


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../../../detail/epilogue.hpp"

