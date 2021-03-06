#include <cassert>
#include <cusend/lazy/scheduler/detail/detail/indirect_receiver_as_invocable.hpp>


namespace ns = cusend;


#ifndef __host__
#define __host__
#define __device__
#endif


struct my_copyable_receiver
{
  int& result;

  __host__ __device__
  void set_value(int value) noexcept
  {
    result = value;
  }

  void set_error(std::exception_ptr) {}

  __host__ __device__
  void set_done() noexcept {}
};


struct my_move_only_receiver
{
  int& result;

  my_move_only_receiver(my_move_only_receiver&&) = default;

  __host__ __device__
  void set_value(int value) noexcept
  {
    result = value;
  }

  void set_error(std::exception_ptr) {}

  __host__ __device__
  void set_done() noexcept {}
};


__host__ __device__
void test()
{
  using namespace cusend::detail;

  {
    // test move-only receiver
    int result = 0;
    int expected = 13;

    my_move_only_receiver r{result};

    auto f = indirectly_as_invocable(&r);

    // test move ctor
    auto g = std::move(f);

    g(expected);

    assert(expected == result);
  }

  {
    // test copyable receiver
    int result = 0;
    int expected = 13;

    my_copyable_receiver r{result};

    auto f = indirectly_as_invocable(&r);

    // test copy ctor
    auto g = f;

    g(expected);

    assert(expected == result);
  }
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

void test_indirect_receiver_as_invocable()
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

