#include <cassert>
#include <cstring>
#include <cusend/execution/executor/inline_executor.hpp>
#include <cusend/just.hpp>
#include <cusend/just_on.hpp>
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


template<class Executor>
__host__ __device__
void test_is_typed_sender(Executor ex)
{
  {
    auto result = ns::just_on(ex);
    static_assert(ns::is_typed_sender<decltype(result)>::value, "Error.");
  }

  {
    auto result = ns::just_on(ex,1);
    static_assert(ns::is_typed_sender<decltype(result)>::value, "Error.");
  }

  {
    auto result = ns::just_on(ex,1,2);
    static_assert(ns::is_typed_sender<decltype(result)>::value, "Error.");
  }

  {
    auto result = ns::just_on(ex,1,2,3);
    static_assert(ns::is_typed_sender<decltype(result)>::value, "Error.");
  }
}


template<class Executor>
__host__ __device__
void test_copyable(Executor ex)
{
  result1 = 0;
  int expected = 13;

  my_receiver r;

  ns::just_on(ex, expected).connect(std::move(r)).start();

  assert(expected == result1);
}


template<class Executor>
__host__ __device__
void test_move_only(Executor ex)
{
  result1 = 0;
  int expected = 13;

  my_receiver r;

  ns::just_on(ex, move_only{expected}).connect(std::move(r)).start();

  assert(expected == result1);
}


template<class Executor>
__host__ __device__
void test_variadic(Executor ex)
{
  int expected1 = 13;
  int expected2 = 7;
  int expected3 = 42;

  {
    result1 = 0;

    my_receiver r;

    ns::just_on(ex).connect(std::move(r)).start();

    assert(true == result1);
  }

  {
    result1 = 0;

    my_receiver r;

    ns::just_on(ex, expected1).connect(std::move(r)).start();

    assert(expected1 == result1);
  }

  {
    result1 = 0;
    result2 = 0;

    my_receiver r;

    ns::just_on(ex, expected1, expected2).connect(std::move(r)).start();

    assert(expected1 == result1);
    assert(expected2 == result2);
  }

  {
    result1 = 0;
    result2 = 0;
    result3 = 0;

    my_receiver r;

    ns::just_on(ex, expected1, expected2, expected3).connect(std::move(r)).start();

    assert(expected1 == result1);
    assert(expected2 == result2);
    assert(expected3 == result3);
  }
}


struct my_executor_with_just_on_member_function : ns::execution::inline_executor
{
  template<class... Types>
  __host__ __device__
  auto just_on(Types&&... values) const
    -> decltype(ns::just_on(ns::execution::inline_executor(), std::forward<Types>(values)...))
  {
    return ns::just_on(ns::execution::inline_executor(), std::forward<Types>(values)...);
  }
};


struct my_executor_with_just_on_free_function : ns::execution::inline_executor {};

template<class... Types>
__host__ __device__
auto just_on(my_executor_with_just_on_free_function, Types&&... values)
  -> decltype(ns::just_on(ns::execution::inline_executor{}, std::forward<Types>(values)...))
{
  return ns::just_on(ns::execution::inline_executor{}, std::forward<Types>(values)...);
}


struct my_scheduler
{
  __host__ __device__
  ns::just_t<> schedule() const
  {
    return ns::just();
  }

  bool operator==(const my_scheduler&) const;
  bool operator!=(const my_scheduler&) const;
};


struct my_scheduler_with_just_on_member_function
{
  ns::just_t<> schedule() const;

  bool operator==(const my_scheduler_with_just_on_member_function&) const;
  bool operator!=(const my_scheduler_with_just_on_member_function&) const;

  template<class... Values>
  __host__ __device__
  ns::just_t<Values&&...> just_on(Values&&... values) const
  {
    return ns::just(std::forward<Values>(values)...);
  }
};


struct my_scheduler_with_just_on_free_function
{
  ns::just_t<> schedule() const;

  bool operator==(const my_scheduler_with_just_on_free_function&) const;
  bool operator!=(const my_scheduler_with_just_on_free_function&) const;
};

template<class... Values>
__host__ __device__
ns::just_t<Values...> just_on(const my_scheduler_with_just_on_free_function, Values&&... values)
{
  return ns::just(std::forward<Values>(values)...);
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


struct gpu_executor
{
  __host__ __device__
  bool operator==(const gpu_executor&) const { return true; }

  __host__ __device__
  bool operator!=(const gpu_executor&) const { return false; }

  template<class Function>
  __host__ __device__
  void execute(Function f) const noexcept
  {
    device_invoke(f);
  }
};


void test_just_on()
{
  test_is_typed_sender(ns::execution::inline_executor{});
  test_is_typed_sender(my_executor_with_just_on_member_function{});
  test_is_typed_sender(my_executor_with_just_on_free_function{});
  test_is_typed_sender(my_scheduler{});
  test_is_typed_sender(my_scheduler_with_just_on_member_function{});
  test_is_typed_sender(my_scheduler_with_just_on_free_function{});

  test_copyable(ns::execution::inline_executor{});
  test_copyable(my_executor_with_just_on_member_function{});
  test_copyable(my_executor_with_just_on_free_function{});
  test_copyable(my_scheduler{});
  test_copyable(my_scheduler_with_just_on_member_function{});
  test_copyable(my_scheduler_with_just_on_free_function{});

  test_move_only(ns::execution::inline_executor{});
  test_move_only(my_executor_with_just_on_member_function{});
  test_move_only(my_executor_with_just_on_free_function{});
  test_move_only(my_scheduler{});
  test_move_only(my_scheduler_with_just_on_member_function{});
  test_move_only(my_scheduler_with_just_on_free_function{});

  test_variadic(ns::execution::inline_executor{});
  test_variadic(my_executor_with_just_on_member_function{});
  test_variadic(my_executor_with_just_on_free_function{});
  test_variadic(my_scheduler{});
  test_variadic(my_scheduler_with_just_on_member_function{});
  test_variadic(my_scheduler_with_just_on_free_function{});

#ifdef __CUDACC__
  test_is_typed_sender(gpu_executor{});
  test_copyable(gpu_executor{});
  test_variadic(gpu_executor{});

  device_invoke([] __device__ ()
  {
    test_is_typed_sender(ns::execution::inline_executor{});
    test_is_typed_sender(my_executor_with_just_on_member_function{});
    test_is_typed_sender(my_executor_with_just_on_free_function{});
    test_is_typed_sender(my_scheduler{});
    test_is_typed_sender(my_scheduler_with_just_on_member_function{});
    test_is_typed_sender(my_scheduler_with_just_on_free_function{});
    test_is_typed_sender(gpu_executor{});

    test_copyable(ns::execution::inline_executor{});
    test_copyable(my_executor_with_just_on_member_function{});
    test_copyable(my_executor_with_just_on_free_function{});
    test_copyable(my_scheduler{});
    test_copyable(my_scheduler_with_just_on_member_function{});
    test_copyable(my_scheduler_with_just_on_free_function{});
    test_copyable(gpu_executor{});

    test_move_only(ns::execution::inline_executor{});
    test_move_only(my_executor_with_just_on_member_function{});
    test_move_only(my_executor_with_just_on_free_function{});
    test_move_only(my_scheduler{});
    test_move_only(my_scheduler_with_just_on_member_function{});
    test_move_only(my_scheduler_with_just_on_free_function{});
    // XXX note we don't test move only with gpu_executor because that's not possible

    test_variadic(ns::execution::inline_executor{});
    test_variadic(my_executor_with_just_on_member_function{});
    test_variadic(my_executor_with_just_on_free_function{});
    test_variadic(my_scheduler{});
    test_variadic(my_scheduler_with_just_on_member_function{});
    test_variadic(my_scheduler_with_just_on_free_function{});
    test_variadic(gpu_executor{});
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

