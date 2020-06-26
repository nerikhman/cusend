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

#include "../../../detail/prologue.hpp"

#include <exception>
#include <utility>
#include "../../../detail/type_traits.hpp"
#include "../../receiver/is_receiver.hpp"
#include "../../receiver/is_receiver_of.hpp"
#include "../../receiver/set_done.hpp"
#include "../../receiver/set_error.hpp"
#include "../../receiver/set_value.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class R>
class receiver_as_invocable
{
  private:
    static_assert(is_receiver<R>::value, "receiver_as_invocable<R>: R is not a receiver");

    // XXX better to replace these members with a __host__ __device__ optional<R> type when available
    //     once we do that, we can eliminate the CUSEND_EXEC_CHECK_DISABLE below
    R r_;
    bool valid_;


  public:
    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    explicit receiver_as_invocable(R&& r) noexcept
      : r_{std::move(r)}, valid_{true}
    {}


    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    receiver_as_invocable(receiver_as_invocable&& other) noexcept
      : r_{std::move(other.r_)}, valid_(other.valid_)
    {
      other.valid_ = false;
    }


    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    receiver_as_invocable(const receiver_as_invocable& other) noexcept
      : r_{other.r_}, valid_{other.valid_}
    {}


    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    ~receiver_as_invocable() noexcept
    {
      if(valid_)
      {
        CUSEND_NAMESPACE::set_done(std::move(r_));
      }
    }


#if CUSEND_HAS_EXCEPTIONS
    template<class... Args,
             CUSEND_REQUIRES(is_receiver_of<R, Args&&...>::value)
            >
    CUSEND_ANNOTATION
    void operator()(Args&&... args)
    {
      try
      {
        CUSEND_NAMESPACE::set_value(std::move(r_), std::forward<Args>(args)...);
      }
      catch(...)
      {
        CUSEND_NAMESPACE::set_error(std::move(r_), std::current_exception());
      }

      valid_ = false;
    }
#else
    template<class... Args,
             CUSEND_REQUIRES(is_receiver_of<R, Args&&...>::value)
            >
    CUSEND_ANNOTATION
    void operator()(Args&&... args)
    {
      CUSEND_NAMESPACE::set_value(std::move(r_), std::forward<Args>(args)...);
      valid_ = false;
    }
#endif
};


template<class R,
         CUSEND_REQUIRES(is_receiver<R>::value)
        >
CUSEND_ANNOTATION
receiver_as_invocable<R> move_as_invocable(R& r)
{
  return receiver_as_invocable<R>{std::move(r)};
}


template<class R,
         CUSEND_REQUIRES(is_receiver<R>::value)
        >
CUSEND_ANNOTATION
receiver_as_invocable<R> copy_as_invocable(const R& r)
{
  R r_copy = r;
  return detail::move_as_invocable(r_copy);
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"

