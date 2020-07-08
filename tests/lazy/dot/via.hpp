#include <cassert>
#include <cstring>
#include <cusend/execution/executor/stream_executor.hpp>
#include <cusend/lazy/as_scheduler.hpp>
#include <cusend/lazy/device_scheduler.hpp>
#include <cusend/lazy/dot/via.hpp>
#include <cusend/lazy/just.hpp>
#include <cusend/lazy/transform.hpp>


namespace ns = cusend;


#ifndef __host__
#define __host__
#define __device__
#define __managed__
#define __global__
#endif


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


__managed__ int num_calls_to_customizations = 0;

struct my_scheduler
{
  __host__ __device__
  bool operator==(const my_scheduler&) const { return true; }

  __host__ __device__
  bool operator!=(const my_scheduler&) const { return false; }

  __host__ __device__
  ns::just_t<> schedule() const noexcept
  {
    ++num_calls_to_customizations;
    return ns::just();
  }
};


struct my_gpu_scheduler
{
  __host__ __device__
  bool operator==(const my_gpu_scheduler&) const { return true; }


  __host__ __device__
  bool operator!=(const my_gpu_scheduler&) const { return false; }


  struct executor
  {
    __host__ __device__
    bool operator==(const executor&) const { return true; }

    __host__ __device__
    bool operator!=(const executor&) const { return false; }

    template<class Function>
    __host__ __device__
    void execute(Function f) const noexcept
    {
      device_invoke(f);
    }
  };


  __host__ __device__
  auto schedule() const
    -> decltype(ns::schedule(ns::as_scheduler(executor{})))
  {
    ++num_calls_to_customizations;
    return ns::schedule(ns::as_scheduler(executor{}));
  }
};


struct my_scheduler_with_via_free_function
{
  __host__ __device__
  ns::just_t<> schedule() const
  {
    return ns::just();
  }

  bool operator==(const my_scheduler_with_via_free_function&) const;
  bool operator!=(const my_scheduler_with_via_free_function&) const;

  template<class Sender>
  __host__ __device__
  friend auto via(Sender&& sender, const my_scheduler_with_via_free_function&)
    -> decltype(ns::dot::via(std::forward<Sender>(sender), ns::as_scheduler(ns::execution::inline_executor{})))
  {
    ++num_calls_to_customizations;
    return ns::dot::via(std::forward<Sender>(sender), ns::as_scheduler(ns::execution::inline_executor{}));
  }
};


__managed__ int result1;
__managed__ int result2;


struct my_receiver
{
  __host__ __device__
  void set_value() &&
  {
    result1 = true;
  }

  __host__ __device__
  void set_value(int value) &&
  {
    result1 = value;
  }

  __host__ __device__
  void set_value(int value1, int value2) &&
  {
    result1 = value1;
    result2 = value2;
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


template<class Executor>
__host__ __device__
void test(Executor ex)
{
  int arg1 = 13;
  int arg2 = 7;
  int expected = arg1 + arg2;

  result1 = 0;
  num_calls_to_customizations = 0;

  auto sender = ns::transform(ns::dot::via(ns::just(arg1), ex), [=] __host__ __device__ (int arg1)
  {
    return arg1 + arg2;
  });

  static_assert(is_chaining(sender), "Error.");

  ns::submit(std::move(sender), my_receiver{});

  assert(result1 == expected);
  assert(1 == num_calls_to_customizations);
}


void test_via_device_scheduler()
{
  // device_scheduler has a customization for via()

  {
    result1 = false;

    // just().via(scheduler)
    auto sender = ns::dot::via(cusend::just(), ns::device_scheduler<ns::execution::stream_executor>{});

    std::move(sender).connect(my_receiver{}).start();

    cudaStreamSynchronize(0);
    assert(result1);
  }

  {
    result1 = -1;

    // just(13).via(scheduler)
    int expected1 = 13;
    auto sender = ns::dot::via(ns::just(expected1), ns::device_scheduler<ns::execution::stream_executor>{});

    std::move(sender).connect(my_receiver{}).start();

    cudaStreamSynchronize(0);
    assert(expected1 == result1);
  }

  {
    result1 = -1;
    result2 = -1;

    // just(13,7).via(scheduler)
    int expected1 = 13;
    int expected2 = 7;
    auto sender = ns::dot::via(ns::just(expected1,expected2), ns::device_scheduler<ns::execution::stream_executor>{});

    std::move(sender).connect(my_receiver{}).start();

    cudaStreamSynchronize(0);
    assert(expected1 == result1);
    assert(expected2 == result2);
  }
}


void test_via()
{
  test(my_scheduler{});
  test(my_scheduler_with_via_free_function{});

#if __CUDACC__
  test(my_gpu_scheduler{});
  test_via_device_scheduler();

  device_invoke([] __device__ ()
  {
    test(my_scheduler{});
    test(my_scheduler_with_via_free_function{});
    test(my_gpu_scheduler{});
  });
#endif
}

