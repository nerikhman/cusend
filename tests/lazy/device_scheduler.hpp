#include <cassert>
#include <cstring>
#include <cusend/execution/executor/stream_executor.hpp>
#include <cusend/lazy/device_scheduler.hpp>
#include <cusend/lazy/get_executor.hpp>
#include <cusend/lazy/is_device_scheduler.hpp>
#include <cusend/lazy/schedule.hpp>
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
void test_is_device_scheduler()
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
  assert(cudaSuccess == cudaDeviceSynchronize());

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
  assert(cudaSuccess == cudaDeviceSynchronize());

#if defined(__CUDACC__)
  assert(!set_error_called);
  assert(expected == set_value_result);
#else
  assert(set_error_called);
#endif
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
  test_is_device_scheduler<Executor>();
  test_schedule(ex);
  test_via(ex);
}


void test_device_scheduler()
{
  test(ns::execution::stream_executor{});
}

