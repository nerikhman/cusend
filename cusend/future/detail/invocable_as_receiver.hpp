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
#include <utility>
#include "../../detail/functional/invoke.hpp"
#include "../../detail/type_traits/invoke_result.hpp"
#include "../../detail/type_traits/is_invocable.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Invocable>
class invocable_as_receiver
{
  public:
    template<class OtherInvocable,
             CUSEND_REQUIRES(std::is_constructible<Invocable,OtherInvocable&&>::value)
            >
    invocable_as_receiver(OtherInvocable&& invocable)
      : invocable_{std::forward<OtherInvocable>(invocable)}
    {}

    invocable_as_receiver(invocable_as_receiver&&) = default;

    template<class... Args,
             CUSEND_REQUIRES(is_invocable<Invocable&&,Args&&...>::value),
             class Result = invoke_result_t<Invocable&&,Args&&...>
            >
    Result set_value(Args&&... args) &&
    {
      return detail::invoke(std::move(invocable_), std::forward<Args>(args)...);
    }

    template<class E>
    void set_error(E&&) && noexcept
    {
      std::terminate();
    }
  
    void set_done() && noexcept {}

  private:
    Invocable invocable_;
};


template<class Invocable>
invocable_as_receiver<typename std::decay<Invocable>::type> as_receiver(Invocable&& f)
{
  return {std::forward<Invocable>(f)};
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"
