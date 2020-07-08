#include <cassert>
#include <cstring>
#include <cusend/execution/executor/stream_executor.hpp>
#include <cusend/lazy/scheduler/bulk_schedule.hpp>
#include <cusend/lazy/scheduler/device_scheduler.hpp>
#include <cusend/lazy/scheduler/get_executor.hpp>
#include <cusend/lazy/scheduler/is_device_scheduler.hpp>
#include <cusend/lazy/scheduler/schedule.hpp>
#include <cusend/lazy/sender/is_sender.hpp>
#include <cusend/lazy/submit.hpp>
#include <cusend/lazy/via.hpp>


#ifndef __host__
#define __host__
#define __device__
#define __managed__
#define __global__
#endif



namespace ns = cusend;


__managed__ int set_value_result;
__managed__ bool set_error_called;


struct my_receiver
{
  __host__ __device__
  void set_value() &&
  {
    set_value_result = true;
  }

  __host__ __device__
  void set_value(int arg) &&
  {
    set_value_result = arg;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept
  {
    set_error_called = true;
  }

  __host__ __device__
  void set_done() && noexcept {}
};

static_assert(ns::is_receiver_of<my_receiver,int&&>::value, "Error.");


template<class Executor>
void test_is_device_scheduler(Executor)
{
  static_assert(ns::is_device_scheduler<ns::device_scheduler<Executor>>::value, "Error.");
}


template<class Executor>
void test_schedule(Executor ex)
{
  ns::device_scheduler<Executor> scheduler = ns::as_scheduler(ex);

  auto sender = ns::schedule(scheduler);
  static_assert(ns::is_sender<decltype(sender)>::value, "Error.");
  assert(ex == ns::get_executor(sender));

  set_value_result = false;
  set_error_called = false;

  ns::submit(std::move(sender), my_receiver{});
  assert(cudaSuccess == cudaStreamSynchronize(scheduler.executor().stream()));

#if defined(__CUDACC__)
  assert(set_value_result);
#else
  assert(set_error_called);
#endif
}


template<class Executor>
void test_via(Executor ex)
{
  ns::device_scheduler<Executor> scheduler = ns::as_scheduler(ex);

  set_value_result = false;
  set_error_called = false;

  int expected = 13;
  auto just_expected = ns::just(expected);

  ns::submit(ns::via(std::move(just_expected), scheduler), my_receiver{});
  assert(cudaSuccess == cudaStreamSynchronize(scheduler.executor().stream()));

#if defined(__CUDACC__)
  assert(!set_error_called);
  assert(expected == set_value_result);
#else
  assert(set_error_called);
#endif
}


__managed__ bool set_value_result0;
__managed__ bool set_value_result1;


struct my_many_receiver
{
  int expected0;
  int expected1;

  __host__ __device__
  void set_value(std::size_t idx) const
  {
    switch(idx)
    {
      case 0:
      {
        set_value_result0 = true;
        break;
      }

      case 1:
      {
        set_value_result1 = true;
        break;
      }

      default:
      {
        assert(false);
        break;
      }
    }
  }

  __host__ __device__
  void set_value(std::size_t idx, int& value) const
  {
    switch(idx)
    {
      case 0:
      {
        set_value_result0 = (expected0 == value);
        break;
      }

      case 1:
      {
        set_value_result1 = (expected0 == value);
        break;
      }

      default:
      {
        assert(false);
        break;
      }
    }
  }


  __host__ __device__
  void set_value(std::size_t idx, int& value0, int& value1) const
  {
    switch(idx)
    {
      case 0:
      {
        set_value_result0 = (expected0 == value0);
        break;
      }

      case 1:
      {
        set_value_result1 = (expected1 == value1);
        break;
      }

      default:
      {
        assert(false);
        break;
      }
    }
  }


  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept
  {
    set_error_called = true;
  }

  __host__ __device__
  void set_done() && noexcept {}
};


template<class Executor>
void test_bulk_schedule(Executor ex)
{
  ns::device_scheduler<Executor> scheduler = ns::as_scheduler(ex);

  {
    set_value_result0 = false;
    set_value_result1 = false;
    set_error_called = false;

    auto s0 = ns::just();
    auto s1 = ns::bulk_schedule(scheduler, 2, std::move(s0));

    ns::submit(std::move(s1), my_many_receiver{13,7});
    assert(cudaSuccess == cudaStreamSynchronize(scheduler.executor().stream()));

#if defined(__CUDACC__)
    assert(!set_error_called);
    assert(set_value_result0);
    assert(set_value_result1);
#else
    assert(set_error_called);
#endif
  }

  {
    set_value_result0 = false;
    set_value_result1 = false;
    set_error_called = false;

    auto s0 = ns::just(13);
    auto s1 = ns::bulk_schedule(scheduler, 2, std::move(s0));

    ns::submit(std::move(s1), my_many_receiver{13,7});
    assert(cudaSuccess == cudaStreamSynchronize(scheduler.executor().stream()));

#if defined(__CUDACC__)
    assert(!set_error_called);
    assert(set_value_result0);
    assert(set_value_result1);
#else
    assert(set_error_called);
#endif
  }

  {
    set_value_result0 = false;
    set_value_result1 = false;
    set_error_called = false;

    auto s0 = ns::just(13,7);
    auto s1 = ns::bulk_schedule(scheduler, 2, std::move(s0));

    ns::submit(std::move(s1), my_many_receiver{13,7});
    assert(cudaSuccess == cudaStreamSynchronize(scheduler.executor().stream()));

#if defined(__CUDACC__)
    assert(!set_error_called);
    assert(set_value_result0);
    assert(set_value_result1);
#else
    assert(set_error_called);
#endif
  }
}


template<class F>
__global__ void device_invoke_kernel(F f)
{
  f();
}


template<class F>
void device_invoke(F f)
{
#if defined(__CUDACC__)

#if !defined(__CUDA_ARCH__)
  // __host__ path

  device_invoke_kernel<<<1,1>>>(f);

#else
  // __device__ path

  // workaround restriction on parameters with copy ctors passed to triple chevrons
  void* ptr_to_arg = cudaGetParameterBuffer(std::alignment_of<F>::value, sizeof(F));
  std::memcpy(ptr_to_arg, &f, sizeof(F));

  // launch the kernel
  if(cudaLaunchDevice(&device_invoke_kernel<F>, ptr_to_arg, dim3(1), dim3(1), 0, 0) != cudaSuccess)
  {
    assert(0);
  }
#endif

  assert(cudaDeviceSynchronize() == cudaSuccess);
#else
  // device invocations are not supported
  assert(0);
#endif
}


template<class Executor>
void test(Executor ex)
{
  test_is_device_scheduler(ex);
  test_schedule(ex);
  test_via(ex);
  test_bulk_schedule(ex);
}


void test_device_scheduler()
{
  test(ns::execution::stream_executor{});
}

