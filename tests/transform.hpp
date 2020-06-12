#include <cassert>
#include <cstring>
#include <cusend/execution/executor/inline_executor.hpp>
#include <cusend/just.hpp>
#include <cusend/transform.hpp>


namespace ns = cusend;


#ifndef __CUDACC__
#define __host__
#define __device__
#define __managed__
#define __global__
#endif


__managed__ int result;


template<class Function>
class move_only_function
{
  public:
    __host__ __device__
    move_only_function(Function f)
      : f_(f)
    {}

    move_only_function(const move_only_function&) = delete;

    move_only_function(move_only_function&&) = default;

    __host__ __device__
    int operator()(int arg) const
    {
      return f_(arg);
    }

  private:
    Function f_;
};


template<class Function>
__host__ __device__
move_only_function<Function> make_move_only_function(Function f)
{
  return {f};
}


struct my_receiver
{
  __host__ __device__
  void set_value(int value)
  {
    result = value;
  }

  void set_error(std::exception_ptr) {}

  __host__ __device__
  void set_done() noexcept {}
};


__host__ __device__
void test_is_typed_sender()
{
  using namespace ns;

  {
    auto result = transform(just(), [] { return; });
    static_assert(is_typed_sender<decltype(result)>::value, "Error.");
  }

  {
    auto result = transform(just(), [] { return 13; });
    static_assert(is_typed_sender<decltype(result)>::value, "Error.");
  }

  {
    auto result = transform(just(13), [](int arg) { return arg; });
    static_assert(is_typed_sender<decltype(result)>::value, "Error.");
  }

  {
    auto result = transform(just(13,7), [](int arg1, int arg2) { return arg1 + arg2; });
    static_assert(is_typed_sender<decltype(result)>::value, "Error.");
  }
}


__host__ __device__
void test_copyable_continuation()
{
  using namespace ns;

  result = 0;
  int arg1 = 13;
  int arg2 = 7;
  int expected = arg1 + arg2;

  my_receiver r;

  transform(just(arg1), [=] (int arg1) { return arg1 + arg2; }).connect(std::move(r)).start();

  assert(expected == result);
}


__host__ __device__
void test_move_only_continuation()
{
  using namespace ns;

  result = 0;
  int arg1 = 13;
  int arg2 = 7;
  int expected = arg1 + arg2;

  my_receiver r;

  auto continuation = make_move_only_function([=] (int arg1) { return arg1 + arg2; });

  transform(just(arg1), std::move(continuation)).connect(std::move(r)).start();

  assert(expected == result);
}


struct my_sender_with_transform_member_function
{
  int arg;

  template<class Function>
  __host__ __device__
  auto transform(Function continuation) &&
    -> decltype(ns::transform(ns::just(arg), continuation))
  {
    return ns::transform(ns::just(arg), continuation);
  }
};


__host__ __device__
void test_sender_with_transform_member_function()
{
  result = 0;
  int arg1 = 13;
  int arg2 = 7;
  int expected = arg1 + arg2;

  my_receiver r;

  my_sender_with_transform_member_function s{arg1};

  ns::transform(std::move(s), [=](int arg1) {return arg1 + arg2;}).connect(std::move(r)).start();

  assert(expected == result);
}


struct my_sender_with_transform_free_function
{
  int arg;
};


template<class Function>
__host__ __device__
auto transform(my_sender_with_transform_free_function&& s, Function continuation)
  -> decltype(ns::transform(ns::just(s.arg), continuation))
{
  return ns::transform(ns::just(s.arg), continuation);
}


__host__ __device__
void test_sender_with_transform_free_function()
{
  result = 0;
  int arg1 = 13;
  int arg2 = 7;
  int expected = arg1 + arg2;

  my_receiver r;

  my_sender_with_transform_member_function s{arg1};

  ns::transform(std::move(s), [=](int arg1) {return arg1 + arg2;}).connect(std::move(r)).start();

  assert(expected == result);
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


void test_transform()
{
  test_is_typed_sender();
  test_copyable_continuation();
  test_move_only_continuation();
  test_sender_with_transform_member_function();
  test_sender_with_transform_free_function();

#ifdef __CUDACC__
  device_invoke([] __device__ ()
  {
    test_is_typed_sender();
    test_copyable_continuation();
    test_move_only_continuation();
    test_sender_with_transform_member_function();
    test_sender_with_transform_free_function();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

