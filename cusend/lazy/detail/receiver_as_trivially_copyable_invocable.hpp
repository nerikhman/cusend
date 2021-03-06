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
#include <type_traits>
#include <utility>
#include "../receiver/is_receiver.hpp"
#include "../receiver/is_receiver_of.hpp"
#include "../receiver/set_error.hpp"
#include "../receiver/set_value.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class R>
struct receiver_as_trivially_copyable_invocable
{
  R r_;

#if CUSEND_HAS_EXCEPTIONS
    template<class... Args,
             CUSEND_REQUIRES(is_receiver_of<R, Args&&...>::value)
            >
    CUSEND_ANNOTATION
    void operator()(Args&&... args)
    {
      try
      {
        set_value(std::move(r_), std::forward<Args>(args)...);
      }
      catch(...)
      {
        set_error(std::move(r_), std::current_exception());
      }
    }
#else
    template<class... Args,
             CUSEND_REQUIRES(is_receiver_of<R, Args&&...>::value)
            >
    CUSEND_ANNOTATION
    void operator()(Args&&... args)
    {
      set_value(std::move(r_), std::forward<Args>(args)...);
    }
#endif
};


template<class Receiver,
         CUSEND_REQUIRES(std::is_trivially_copyable<Receiver>::value),
         CUSEND_REQUIRES(is_receiver<Receiver>::value)
        >
CUSEND_ANNOTATION
receiver_as_trivially_copyable_invocable<Receiver> as_trivially_copyable_invocable(Receiver receiver)
{
  return {std::forward<Receiver>(receiver)};
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../detail/epilogue.hpp"

