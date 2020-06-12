#include <cassert>
#include <cstring>
#include <cusend/dot/just.hpp>
#include <cusend/sender/is_typed_sender.hpp>
#include <exception>
#include <utility>


namespace ns = cusend;


#ifndef __CUDACC__
#define __host__
#define __device__
#define __managed__
#define __global__
#endif


__managed__ int result1;
__managed__ int result2;
__managed__ int result3;


struct move_only
{
  int value;

  move_only(move_only&&) = default;
};


struct my_receiver
{
  __host__ __device__
  void set_value()
  {
    result1 = true;
  }

  __host__ __device__
  void set_value(int value)
  {
    result1 = value;
  }

  __host__ __device__
  void set_value(int value1, int value2)
  {
    result1 = value1;
    result2 = value2;
  }

  __host__ __device__
  void set_value(int value1, int value2, int value3)
  {
    result1 = value1;
    result2 = value2;
    result3 = value3;
  }

  __host__ __device__
  void set_value(move_only&& value)
  {
    result1 = value.value;
  }

  void set_error(std::exception_ptr) {}

  __host__ __device__
  void set_done() noexcept {}
};


__host__ __device__
void test_is_typed_sender()
{
  {
    auto result = ns::dot::just();
    static_assert(ns::is_typed_sender<decltype(result)>::value, "Error.");
  }

  {
    auto result = ns::dot::just(1);
    static_assert(ns::is_typed_sender<decltype(result)>::value, "Error.");
  }

  {
    auto result = ns::dot::just(1,2);
    static_assert(ns::is_typed_sender<decltype(result)>::value, "Error.");
  }

  {
    auto result = ns::dot::just(1,2,3);
    static_assert(ns::is_typed_sender<decltype(result)>::value, "Error.");
  }
}


template<class S>
__host__ __device__
constexpr bool is_chaining(const S&) { return false; }

template<class S>
__host__ __device__
constexpr bool is_chaining(const ns::chaining_sender<S>&) { return true; }


__host__ __device__
void test_is_chaining()
{
  {
    auto result = ns::dot::just();
    static_assert(is_chaining(result), "Error.");
  }

  {
    auto result = ns::dot::just(1);
    static_assert(is_chaining(result), "Error.");
  }

  {
    auto result = ns::dot::just(1,2);
    static_assert(is_chaining(result), "Error.");
  }

  {
    auto result = ns::dot::just(1,2,3);
    static_assert(is_chaining(result), "Error.");
  }
}


__host__ __device__
void test_copyable()
{
  result1 = 0;
  int expected = 13;

  my_receiver r;

  ns::dot::just(expected).connect(std::move(r)).start();

  assert(expected == result1);
}


__host__ __device__
void test_move_only()
{
  result1 = 0;
  int expected = 13;

  my_receiver r;

  ns::dot::just(move_only{expected}).connect(std::move(r)).start();

  assert(expected == result1);
}


__host__ __device__
void test_variadic()
{
  int expected1 = 13;
  int expected2 = 7;
  int expected3 = 42;

  {
    result1 = 0;

    my_receiver r;

    ns::dot::just().connect(std::move(r)).start();

    assert(true == result1);
  }

  {
    result1 = 0;

    my_receiver r;

    ns::dot::just(expected1).connect(std::move(r)).start();

    assert(expected1 == result1);
  }

  {
    result1 = 0;
    result2 = 0;

    my_receiver r;

    ns::dot::just(expected1, expected2).connect(std::move(r)).start();

    assert(expected1 == result1);
    assert(expected2 == result2);
  }

  {
    result1 = 0;
    result2 = 0;
    result3 = 0;

    my_receiver r;

    ns::dot::just(expected1, expected2, expected3).connect(std::move(r)).start();

    assert(expected1 == result1);
    assert(expected2 == result2);
    assert(expected3 == result3);
  }
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


void test_just()
{
  test_is_typed_sender();
  test_is_chaining();
  test_copyable();
  test_move_only();
  test_variadic();

#ifdef __CUDACC__
  device_invoke([] __device__ ()
  {
    test_is_typed_sender();
    test_is_chaining();
    test_copyable();
    test_move_only();
    test_variadic();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

