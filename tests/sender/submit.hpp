#include <cassert>
#include <cudex/sender/set_error.hpp>
#include <cudex/sender/set_done.hpp>
#include <cudex/sender/set_value.hpp>
#include <cudex/sender/submit.hpp>


#ifndef __CUDACC__
#define __host__
#define __device__
#endif


enum class send_to_channel
{
  value,
  error,
  done
};

struct has_submit_member_function
{
  send_to_channel channel;

  template<class R>
  __host__ __device__
  void submit(R&& r) &&
  {
    switch(channel)
    {
      case send_to_channel::value:
      {
        cudex::set_value(std::forward<R>(r));
        break;
      }

      case send_to_channel::error:
      {
        cudex::set_error(std::forward<R>(r), 13);
        break;
      }

      case send_to_channel::done:
      {
        cudex::set_done(std::forward<R>(r));
        break;
      }
    }
  }
};


struct has_submit_free_function
{
  send_to_channel channel;
};

template<class R>
__host__ __device__
void submit(has_submit_free_function&& s, R&& r)
{
  has_submit_member_function ss{s.channel};

  cudex::submit(std::move(ss), std::forward<R>(r));
}


struct my_receiver
{
  bool& set_value_invoked;
  bool& set_error_invoked;
  bool& set_done_invoked;

  __host__ __device__
  void set_value() &&
  {
    set_value_invoked = true;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept
  {
    set_error_invoked = true;
  }

  __host__ __device__
  void set_done() && noexcept
  {
    set_done_invoked = true;
  }
};


__host__ __device__
void test()
{
  {
    // test sender with member function

    {
      // test normal case
      has_submit_member_function s{send_to_channel::value};

      bool set_value_invoked = false;
      bool set_error_invoked = false;
      bool set_done_invoked = false;
      my_receiver r{set_value_invoked, set_error_invoked, set_done_invoked};

      cudex::submit(std::move(s), std::move(r));
      assert(set_value_invoked);
    }

    {
      // test error case
      has_submit_member_function s{send_to_channel::error};

      bool set_value_invoked = false;
      bool set_error_invoked = false;
      bool set_done_invoked = false;
      my_receiver r{set_value_invoked, set_error_invoked, set_done_invoked};

      cudex::submit(std::move(s), std::move(r));
      assert(set_error_invoked);
    }

    {
      // test done case
      has_submit_member_function s{send_to_channel::done};

      bool set_value_invoked = false;
      bool set_error_invoked = false;
      bool set_done_invoked = false;
      my_receiver r{set_value_invoked, set_error_invoked, set_done_invoked};

      cudex::submit(std::move(s), std::move(r));
      assert(set_done_invoked);
    }
  }

  {
    // test sender with free function
    
    {
      // test normal case
      has_submit_free_function s{send_to_channel::value};

      bool set_value_invoked = false;
      bool set_error_invoked = false;
      bool set_done_invoked = false;
      my_receiver r{set_value_invoked, set_error_invoked, set_done_invoked};

      cudex::submit(std::move(s), std::move(r));
      assert(set_value_invoked);
    }

    {
      // test error case
      has_submit_free_function s{send_to_channel::error};

      bool set_value_invoked = false;
      bool set_error_invoked = false;
      bool set_done_invoked = false;
      my_receiver r{set_value_invoked, set_error_invoked, set_done_invoked};

      cudex::submit(std::move(s), std::move(r));
      assert(set_error_invoked);
    }

    {
      // test done case
      has_submit_free_function s{send_to_channel::done};

      bool set_value_invoked = false;
      bool set_error_invoked = false;
      bool set_done_invoked = false;
      my_receiver r{set_value_invoked, set_error_invoked, set_done_invoked};

      cudex::submit(std::move(s), std::move(r));
      assert(set_done_invoked);
    }
  }
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_submit()
{
  test();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] __device__ ()
  {
    test();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

