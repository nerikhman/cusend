#include <cassert>
#include <cstring>
#include <cusend/execution/executor/inline_executor.hpp>
#include <cusend/lazy/dot/schedule.hpp>
#include <cusend/lazy/scheduler/get_executor.hpp>
#include <cusend/lazy/sender/is_sender.hpp>
#include <cusend/lazy/submit.hpp>


#ifndef __host__
#define __host__
#define __device__
#define __managed__
#define __global__
#endif



namespace ns = cusend;


struct my_receiver
{
  bool& received;

  __host__ __device__
  void set_value() &&
  {
    received = true;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


template<class S>
__host__ __device__
constexpr bool is_chaining(const S&) { return false; }

template<class S>
__host__ __device__
constexpr bool is_chaining(const ns::chaining_sender<S>&) { return true; }


__host__ __device__
void test()
{
  auto sender = ns::dot::schedule(ns::execution::inline_executor{});

  static_assert(ns::is_sender<decltype(sender)>::value, "Error.");
  static_assert(is_chaining(sender), "Error.");

  assert(ns::execution::inline_executor{} == ns::get_executor(sender));

  bool result = false;
  ns::submit(std::move(sender), my_receiver{result});
  assert(result);
}


template<class F>
__global__ void device_invoke_kernel(F f)
{
  f();
}


template<class F>
__host__ __device__
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


void test_schedule()
{
  test();

#ifdef __CUDACC__
  device_invoke([] __device__ ()
  {
    test();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

