#include <cassert>
#include <utility>
#include <cusend/future.hpp>
#include <cusend/lazy/submit.hpp>

namespace ns = cusend;

void test_default_construction()
{
  // default construction
  ns::future<int> f0;
  assert(!f0.valid());
  assert(!f0.is_ready());


  ns::future<void> f1;
  assert(!f1.valid());
  assert(!f1.is_ready());
}


void test_move_construction()
{
  // move construction
  ns::future<int> f0 = ns::make_ready_future(13);
  assert(f0.valid());

  ns::future<int> f1 = std::move(f0);
  assert(!f0.valid());
  assert(f1.valid());


  ns::future<void> f2 = ns::make_ready_future();
  assert(f2.valid());

  ns::future<void> f3 = std::move(f2);
  assert(!f2.valid());
  assert(f3.valid());
}


void test_move_assignment()
{
  ns::future<int> f1 = ns::make_ready_future(13);
  assert(f1.valid());

  ns::future<int> f2;
  assert(!f2.valid());

  f2 = std::move(f1);
  assert(!f1.valid());
  assert(f2.valid());


  ns::future<void> f3 = ns::make_ready_future();
  assert(f3.valid());

  ns::future<void> f4;
  assert(!f4.valid());

  f4 = std::move(f3);
  assert(!f3.valid());
  assert(f4.valid());
}


void test_get()
{
  ns::future<int> f1 = ns::make_ready_future(13);
  assert(f1.valid());
  assert(f1.is_ready());

  assert(13 == std::move(f1).get());
  assert(!f1.valid());


  ns::future<void> f2 = ns::make_ready_future();
  assert(f2.valid());
  assert(f2.is_ready());

  std::move(f2).get();
  assert(!f2.valid());
}


// future.then has several different code paths
// that depend on the type of input and output future
// these tests exercise each case of .then


void test_then_void_to_void()
{
  // then void -> void
  auto f1 = ns::make_ready_future();
  assert(f1.valid());

  // use a __host__ __device__ lambda because __device__ lambdas cannot return void safely
  auto f2 = std::move(f1).then([] __host__ __device__ { return; } );

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  std::move(f2).get();
  assert(!f2.valid());
}


void test_then_void_to_int()
{
  // then void -> int
  auto f1 = ns::make_ready_future();

  auto f2 = std::move(f1).then([] __device__ () { return 13; });

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(std::move(f2).get() == 13);
  assert(!f2.valid());
}


void test_then_int_to_void()
{
  // then int -> void
  auto f1 = ns::make_ready_future(7);
  assert(f1.valid());
  assert(f1.is_ready());

  // use a __host__ __device__ lambda because __device__ lambdas cannot return void safely
  auto f2 = std::move(f1).then([] __host__ __device__ (int) { return;} );

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  std::move(f2).get();
  assert(!f2.valid());
}


void test_then_int_to_int()
{
  // then int -> int
  auto f1 = ns::make_ready_future(7);

  auto f2 = std::move(f1).then([] __device__ (int arg) { return arg + 6; });

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(std::move(f2).get() == 13);
  assert(!f2.valid());
}


void test_then_int_to_float()
{
  // then int -> float
  auto f1 = ns::make_ready_future(7);

  // use a __host__ __device__ lambda because __device__ lambdas cannot return float safely
  auto f2 = std::move(f1).then([] __host__ __device__ (int arg) { return static_cast<float>(arg + 6); });

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(std::move(f2).get() == 13.f);
  assert(!f2.valid());
}


__managed__ int single_result;


struct receive_void_and_return_void
{
  int result;

  __device__
  void set_value() && noexcept
  {
    single_result = result;
  }

  template<class E>
  __device__
  void set_error(E&&) && noexcept {}

  __device__
  void set_done() && noexcept {}
};


void test_then_receiver_of_void_and_return_void()
{
  auto f1 = ns::make_ready_future();

  int expected = 13;
  single_result = -1;
  auto f2 = std::move(f1).then(receive_void_and_return_void{expected});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());
  assert(f2.valid());

  std::move(f2).get();
  assert(!f2.valid());
  assert(expected == single_result);
}


struct receive_void_and_return_int
{
  int result;

  __device__
  int set_value() && noexcept
  {
    return result;
  }

  template<class E>
  __device__
  void set_error(E&&) && noexcept {}

  __device__
  void set_done() && noexcept {}
};


void test_then_receiver_of_void_and_return_int()
{
  auto f1 = ns::make_ready_future();

  int expected = 13;
  auto f2 = std::move(f1).then(receive_void_and_return_int{expected});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());
  assert(f2.valid());

  assert(expected == std::move(f2).get());
  assert(!f2.valid());
}


struct receive_int_and_return_void
{
  int result;

  __device__
  void set_value(int) && noexcept
  {
    single_result = result;
  }

  template<class E>
  __device__
  void set_error(E&&) && noexcept {}

  __device__
  void set_done() && noexcept {}
};


void test_then_receiver_of_int_and_return_void()
{
  auto f1 = ns::make_ready_future(13);

  int expected = 13;
  single_result = -1;
  auto f2 = std::move(f1).then(receive_int_and_return_void{expected});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());
  assert(f2.valid());

  std::move(f2).get();
  assert(!f2.valid());
  assert(expected == single_result);
}


struct receive_and_add_one
{
  __device__
  int set_value(int value) && noexcept
  {
    return value + 1;
  }

  template<class E>
  __device__
  void set_error(E&&) && noexcept {}

  __device__
  void set_done() && noexcept {}
};


void test_then_receiver_of_int_and_return_int()
{
  int expected = 13;
  auto f1 = ns::make_ready_future(expected - 1);

  auto f2 = std::move(f1).then(receive_and_add_one{});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());
  assert(f2.valid());

  assert(expected == std::move(f2).get());
  assert(!f2.valid());
}


__managed__ bool bulk_result0;
__managed__ bool bulk_result1;


struct many_receiver_of_int
{
  int expected;

  __device__
  void set_value(int idx, int& value) noexcept
  {
    switch(idx)
    {
      case 0:
      {
        bulk_result0 = (expected == value);
        break;
      }

      case 1:
      {
        bulk_result1 = (expected == value);
        break;
      }

      default:
      {
        assert(0);
        break;
      }
    }
  }

  template<class E>
  __device__
  void set_error(E&&) && noexcept {}

  __device__
  void set_done() && noexcept {}
};


void test_bulk_then_receiver_of_int()
{
  int expected = 7;
  auto f1 = ns::make_ready_future(std::move(expected));

  bulk_result0 = false;
  bulk_result1 = false;

  auto f2 = std::move(f1).bulk_then(many_receiver_of_int{expected}, 2);

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(expected == std::move(f2).get());
  assert(bulk_result0);
  assert(bulk_result1);
}


struct many_receiver_of_void
{
  __device__
  void set_value(int idx) noexcept
  {
    switch(idx)
    {
      case 0:
      {
        bulk_result0 = true;
        break;
      }

      case 1:
      {
        bulk_result1 = true;
        break;
      }

      default:
      {
        assert(0);
        break;
      }
    }
  }

  template<class E>
  __device__
  void set_error(E&&) && noexcept {}

  __device__
  void set_done() && noexcept {}
};


void test_bulk_then_receiver_of_void()
{
  auto f1 = ns::make_ready_future();

  bulk_result0 = false;
  bulk_result1 = false;

  auto f2 = std::move(f1).bulk_then(many_receiver_of_void{}, 2);

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(bulk_result0);
  assert(bulk_result1);
}


void test_bulk_then_void_to_void()
{
  auto f1 = ns::make_ready_future();

  bulk_result0 = false;
  bulk_result1 = false;

  auto f2 = std::move(f1).bulk_then([] __device__ (std::size_t idx)
  {
    switch(idx)
    {
      case 0:
      {
        bulk_result0 = true;
        break;
      }

      case 1:
      {
        bulk_result1 = true;
        break;
      }

      default:
      {
        assert(false);
        break;
      }
    }
  }, 2);

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());
  assert(bulk_result0);
  assert(bulk_result1);
}


void test_bulk_then_int_to_int()
{
  int expected = 13;
  auto f1 = ns::make_ready_future(std::move(expected));

  bulk_result0 = false;
  bulk_result1 = false;

  auto f2 = std::move(f1).bulk_then([expected] __device__ (std::size_t idx, int& value)
  {
    switch(idx)
    {
      case 0:
      {
        bulk_result0 = (expected == value);
        break;
      }

      case 1:
      {
        bulk_result1 = (expected == value);
        break;
      }

      default:
      {
        assert(false);
        break;
      }
    }
  }, 2);

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());
  assert(expected == std::move(f2).get());
  assert(bulk_result0);
  assert(bulk_result1);
}


void test_bulk()
{
  {
    // test void future

    bulk_result0 = false;
    bulk_result1 = false;

    ns::submit(ns::make_ready_future().bulk(2), many_receiver_of_void{});
    assert(cudaSuccess == cudaDeviceSynchronize());

    assert(bulk_result0);
    assert(bulk_result1);
  }

  {
    // test int future

    bulk_result0 = false;
    bulk_result1 = false;

    ns::submit(ns::make_ready_future(13).bulk(2), many_receiver_of_int{13});
    assert(cudaSuccess == cudaDeviceSynchronize());

    assert(bulk_result0);
    assert(bulk_result1);
  }
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_future()
{
  test_default_construction();
  test_move_construction();
  test_move_assignment();
  test_get();

#ifdef __CUDACC__
  // these tests require CUDA C++
  test_then_void_to_void();
  test_then_void_to_int();
  test_then_int_to_void();
  test_then_int_to_int();
  test_then_int_to_float();

  test_then_receiver_of_void_and_return_void();
  test_then_receiver_of_void_and_return_int();
  test_then_receiver_of_int_and_return_void();
  test_then_receiver_of_int_and_return_int();

  test_bulk_then_receiver_of_void();
  test_bulk_then_receiver_of_int();

  test_bulk_then_void_to_void();
  test_bulk_then_int_to_int();

  test_bulk();
#endif
}

