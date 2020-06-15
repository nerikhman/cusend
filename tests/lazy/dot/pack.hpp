#include <cassert>
#include <cstring>
#include <cusend/lazy/just.hpp>
#include <cusend/lazy/dot/pack.hpp>
#include <exception>
#include <utility>


namespace ns = cusend;


#ifndef __host__
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


struct my_tuple_receiver
{
  __host__ __device__
  void set_value(ns::detail::tuple<>&& value)
  {
    result1 = true;
  }

  __host__ __device__
  void set_value(ns::detail::tuple<int>&& value)
  {
    result1 = ns::detail::get<0>(value);
  }

  __host__ __device__
  void set_value(ns::detail::tuple<int,int>&& value)
  {
    result1 = ns::detail::get<0>(value);
    result2 = ns::detail::get<1>(value);
  }

  __host__ __device__
  void set_value(ns::detail::tuple<int,int,int>&& value)
  {
    result1 = ns::detail::get<0>(value);
    result2 = ns::detail::get<1>(value);
    result3 = ns::detail::get<2>(value);
  }

  __host__ __device__
  void set_value(ns::detail::tuple<move_only>&& value)
  {
    result1 = ns::detail::get<0>(value).value;
  }

  void set_error(std::exception_ptr) {}

  __host__ __device__
  void set_done() noexcept {}
};


__host__ __device__
void test_is_typed_sender()
{
  {
    auto result = ns::dot::pack(ns::just());
    static_assert(ns::is_typed_sender<decltype(result)>::value, "Error.");
  }

  {
    auto result = ns::dot::pack(ns::just(1));
    static_assert(ns::is_typed_sender<decltype(result)>::value, "Error.");
  }

  {
    auto result = ns::dot::pack(ns::just(1,2));
    static_assert(ns::is_typed_sender<decltype(result)>::value, "Error.");
  }

  {
    auto result = ns::dot::pack(ns::just(1,2,3));
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
    auto result = ns::dot::pack(ns::just());
    static_assert(is_chaining(result), "Error.");
  }

  {
    auto result = ns::dot::pack(ns::just(1));
    static_assert(is_chaining(result), "Error.");
  }

  {
    auto result = ns::dot::pack(ns::just(1,2));
    static_assert(is_chaining(result), "Error.");
  }

  {
    auto result = ns::dot::pack(ns::just(1,2,3));
    static_assert(is_chaining(result), "Error.");
  }
}


__host__ __device__
void test_move_only()
{
  result1 = 0;
  int expected = 13;

  my_tuple_receiver r;

  ns::dot::pack(ns::just(move_only{expected})).connect(std::move(r)).start();

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

    my_tuple_receiver r;

    ns::dot::pack(ns::just()).connect(std::move(r)).start();

    assert(true == result1);
  }

  {
    result1 = 0;

    my_tuple_receiver r;

    ns::dot::pack(ns::just(expected1)).connect(std::move(r)).start();

    assert(expected1 == result1);
  }

  {
    result1 = 0;
    result2 = 0;

    my_tuple_receiver r;

    ns::dot::pack(ns::just(expected1, expected2)).connect(std::move(r)).start();

    assert(expected1 == result1);
    assert(expected2 == result2);
  }

  {
    result1 = 0;
    result2 = 0;
    result3 = 0;

    my_tuple_receiver r;

    ns::dot::pack(ns::just(expected1, expected2, expected3)).connect(std::move(r)).start();

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


void test_pack()
{
  test_is_typed_sender();
  test_is_chaining();
  test_move_only();
  test_variadic();

#ifdef __CUDACC__
  device_invoke([] __device__ ()
  {
    test_is_typed_sender();
    test_move_only();
    test_variadic();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

