#include <cassert>
#include <cudex/sender/connect.hpp>


#ifndef __CUDACC__
#define __host__
#define __device__
#endif


struct my_receiver
{
  __host__ __device__
  void set_value() {}

  template<class E>
  __host__ __device__
  void set_error(E&&) {}

  __host__ __device__
  void set_done() {}
};


struct my_operation
{
  constexpr static bool connected = true;
};


struct has_connect_member_function
{
  template<class R>
  __host__ __device__
  my_operation connect(R&&)
  {
    return {};
  }
};

struct has_connect_free_function {};

template<class R>
__host__ __device__
my_operation connect(has_connect_free_function&, R&& r)
{
  return {};
}


__host__ __device__
void test()
{
  {
    // test sender with member function
    has_connect_member_function s;
    auto operation = cudex::connect(s, my_receiver{});

    assert(operation.connected);
  }

  {
    // test sender with free function
    has_connect_free_function s;
    auto operation = cudex::connect(s, my_receiver{});

    assert(operation.connected);
  }
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_connect()
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

