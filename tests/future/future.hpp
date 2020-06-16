#include <cassert>
#include <utility>
#include <cusend/future.hpp>

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

  try
  {
    auto f2 = std::move(f1).then([] __host__ __device__ { return; } );

#if !defined(__CUDACC__)
    assert(false);
#endif

    assert(!f1.valid());
    assert(f2.valid());
    f2.wait();

    assert(f2.is_ready());

    std::move(f2).get();
    assert(!f2.valid());
  }
  catch(std::runtime_error)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
  }
}


void test_then_int_to_void()
{
  // then int -> void
  auto f1 = ns::make_ready_future(7);
  assert(f1.valid());
  assert(f1.is_ready());

  try
  {
    auto f2 = std::move(f1).then([] __host__ __device__ (int) { return;} );

#if !defined(__CUDACC__)
    assert(false);
#endif

    assert(!f1.valid());
    assert(f2.valid());
    f2.wait();

    assert(f2.is_ready());

    std::move(f2).get();
    assert(!f2.valid());
  }
  catch(std::runtime_error)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
  }
}


void test_then_void_to_int()
{
  // then void -> int
  auto f1 = ns::make_ready_future();

  try
  {
    auto f2 = std::move(f1).then([] __host__ __device__ () { return 13; });

#if !defined(__CUDACC__)
    assert(false);
#endif

    assert(!f1.valid());
    assert(f2.valid());
    f2.wait();

    assert(f2.is_ready());

    assert(std::move(f2).get() == 13);
    assert(!f2.valid());
  }
  catch(std::runtime_error)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
  }
}


void test_then_int_to_int()
{
  // then int -> int
  auto f1 = ns::make_ready_future(7);

  try
  {
    auto f2 = std::move(f1).then([] __host__ __device__ (int arg) { return arg + 6; });

#if !defined(__CUDACC__)
    assert(false);
#endif

    assert(!f1.valid());
    assert(f2.valid());
    f2.wait();

    assert(f2.is_ready());

    assert(std::move(f2).get() == 13);
    assert(!f2.valid());
  }
  catch(std::runtime_error)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
  }
}


void test_then_int_to_float()
{
  // then int -> float
  auto f1 = ns::make_ready_future(7);

  try
  {
    auto f2 = std::move(f1).then([] __host__ __device__ (int arg) { return static_cast<float>(arg + 6); });

#if !defined(__CUDACC__)
    assert(false);
#endif

    assert(!f1.valid());
    assert(f2.valid());
    f2.wait();

    assert(f2.is_ready());

    assert(std::move(f2).get() == 13.f);
    assert(!f2.valid());
  }
  catch(std::runtime_error)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
  }
}


__managed__ int result;


struct receive_and_add_one
{
  __host__ __device__
  int set_value(int value) && noexcept
  {
    return value + 1;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


void test_then_receiver()
{
  int expected = 13;
  auto f1 = ns::make_ready_future(std::move(expected));

  try
  {
    auto f2 = std::move(f1).then(receive_and_add_one{});

#if !defined(__CUDACC__)
    assert(false);
#endif

    assert(!f1.valid());
    assert(f2.valid());
    f2.wait();

    assert(f2.is_ready());
    assert(f2.valid());

    assert(expected + 1 == std::move(f2).get());
    assert(!f2.valid());
  }
  catch(std::runtime_error)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
  }
}


__managed__ bool bulk_result0;
__managed__ bool bulk_result1;


struct many_receiver_of_int
{
  int expected;

  __host__ __device__
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
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


void test_bulk_then_receiver_of_int()
{
  int expected = 7;
  auto f1 = ns::make_ready_future(std::move(expected));

  try
  {
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
  catch(std::runtime_error)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
  }
}


struct many_receiver_of_void
{
  __host__ __device__
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
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


void test_bulk_then_receiver_of_void()
{
  auto f1 = ns::make_ready_future();

  try
  {
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
  catch(std::runtime_error)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
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
  test_then_void_to_void();
  test_then_int_to_void();
  test_then_void_to_int();
  test_then_int_to_int();
  test_then_int_to_float();
  test_then_receiver();
  test_bulk_then_receiver_of_int();
  test_bulk_then_receiver_of_void();
}

