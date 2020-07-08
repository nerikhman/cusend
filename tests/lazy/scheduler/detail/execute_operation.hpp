#include <cassert>
#include <cusend/execution/executor/inline_executor.hpp>
#include <cusend/lazy/scheduler/detail/execute_operation.hpp>

#ifndef __host__
#define __host__
#define __device__
#endif


struct mutable_receiver
{
  int& result;
  int value;

  __host__ __device__
  void set_value()
  {
    result = value;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


struct const_receiver
{
  int& result;
  int value;

  __host__ __device__
  void set_value() const
  {
    result = value;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() noexcept {}
};


struct rvalue_receiver
{
  int& result;
  int value;

  __host__ __device__
  void set_value() &&
  {
    result = value;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() noexcept {}
};


struct copyable_receiver
{
  int& result;
  int value;

  copyable_receiver(const copyable_receiver&) = default;

  __host__ __device__
  void set_value()
  {
    result = value;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() noexcept {}
};


struct move_only_receiver
{
  int& result;
  int value;

  move_only_receiver(move_only_receiver&&) = default;

  __host__ __device__
  void set_value()
  {
    result = value;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() noexcept {}
};


__host__ __device__
void test()
{
  using namespace cusend::execution;
  using namespace cusend::detail;

  inline_executor ex;

  {
    int result = 0;
    int expected = 13;

    mutable_receiver r{result, expected};
    auto op = make_execute_operation(ex, r);

    op.start();

    assert(expected == result);
  }

  {
    int result = 0;
    int expected = 13;

    const_receiver r{result, expected};
    auto op = make_execute_operation(ex, r);

    op.start();

    assert(expected == result);
  }

  {
    int result = 0;
    int expected = 13;

    rvalue_receiver r{result, expected};
    auto op = make_execute_operation(ex, r);

    op.start();

    assert(expected == result);
  }

  {
    int result = 0;
    int expected = 13;

    copyable_receiver r{result, expected};
    auto op = make_execute_operation(ex, r);

    auto op_copy = op;

    op_copy.start();

    assert(expected == result);
  }

  {
    int result = 0;
    int expected = 13;

    move_only_receiver r{result, expected};
    auto op = make_execute_operation(ex, std::move(r));

    auto op_moved = std::move(op);

    op_moved.start();

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


void test_execute_operation()
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

