#include <cassert>
#include <utility>
#include <cusend/future/host_promise.hpp>
#include <cusend/lazy/submit.hpp>

namespace ns = cusend;

template<class Executor>
void test_move_construction(Executor ex)
{
  {
    ns::host_promise<int> p0{};
    ns::detail::host_future<int,Executor> f0 = p0.get_future(ex);
    assert(f0.valid());

    ns::detail::host_future<int,Executor> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }

  {
    ns::host_promise<void> p0;
    ns::detail::host_future<void,Executor> f0 = p0.get_future(ex);
    assert(f0.valid());

    ns::detail::host_future<void,Executor> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }
}


template<class Executor>
void test_move_assignment(Executor ex)
{
  {
    ns::host_promise<int> p0{};
    ns::detail::host_future<int,Executor> f0 = p0.get_future(ex);
    assert(f0.valid());

    ns::detail::host_future<int,Executor> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }


  {
    ns::host_promise<void> p0{};
    ns::detail::host_future<void,Executor> f0 = p0.get_future(ex);
    assert(f0.valid());

    ns::detail::host_future<void,Executor> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }
}


template<class Executor, class T>
ns::detail::host_future<T,Executor> make_ready_host_future(const Executor& ex, T value)
{
  ns::host_promise<T> p{};
  std::move(p).set_value(value);
  return p.get_future(ex);
}


template<class Executor>
ns::detail::host_future<void,Executor> make_ready_host_future(const Executor& ex)
{
  ns::host_promise<void> p{};
  std::move(p).set_value();
  return p.get_future(ex);
}


template<class Executor>
void test_get(Executor ex)
{
  {
    ns::detail::host_future<int,Executor> f = make_ready_host_future(ex, 13);
    assert(f.valid());

    assert(13 == std::move(f).get());
    assert(!f.valid());
  }

  {
    ns::detail::host_future<void,Executor> f = make_ready_host_future(ex);
    assert(f.valid());

    std::move(f).get();
    assert(!f.valid());
  }
}


// future.then has several different code paths
// that depend on the type of input and output future
// these tests exercise each case of .then


template<class StreamExecutor>
void test_then_void_to_void(StreamExecutor ex)
{
  // then void -> void
  auto f1 = make_ready_host_future(ex);
  assert(f1.valid());

  auto f2 = std::move(f1).then(ex, [] __host__ __device__ { return; });

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  std::move(f2).get();
  assert(!f2.valid());
}


template<class StreamExecutor>
void test_then_int_to_void(StreamExecutor ex)
{
  // then int -> void
  auto f1 = make_ready_host_future(ex, 7);
  assert(f1.valid());

  auto f2 = std::move(f1).then(ex, [] __host__ __device__ (int){return;});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  std::move(f2).get();
  assert(!f2.valid());
}


template<class StreamExecutor>
void test_then_void_to_int(StreamExecutor ex)
{
  // then void -> int
  auto f1 = make_ready_host_future(ex);

  auto f2 = std::move(f1).then(ex, [] __host__ __device__ {return 13;});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(std::move(f2).get() == 13);
  assert(!f2.valid());
}


template<class StreamExecutor>
void test_then_int_to_int(StreamExecutor ex)
{
  // then int -> int
  auto f1 = make_ready_host_future(ex, 7);

  auto f2 = std::move(f1).then(ex, [] __host__ __device__ (int arg) { return arg + 6; });

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(std::move(f2).get() == 13);
  assert(!f2.valid());
}


template<class StreamExecutor>
void test_then_int_to_float(StreamExecutor ex)
{
  // then int -> float
  auto f1 = make_ready_host_future(ex, 7);

  auto f2 = std::move(f1).then(ex, [] __host__ __device__ (int arg){return static_cast<float>(arg + 6);});

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

  __host__ __device__
  void set_value() && noexcept
  {
    single_result = result;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


template<class Executor>
void test_then_receiver_of_void_and_return_void(Executor ex)
{
  auto f1 = make_ready_host_future(ex);

  int expected = 13;
  single_result = -1;
  auto f2 = std::move(f1).then(ex, receive_void_and_return_void{expected});

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

  __host__ __device__
  int set_value() && noexcept
  {
    return result;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


template<class Executor>
void test_then_receiver_of_void_and_return_int(Executor ex)
{
  auto f1 = make_ready_host_future(ex);

  int expected = 13;
  auto f2 = std::move(f1).then(ex, receive_void_and_return_int{expected});

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

  __host__ __device__
  void set_value(int) && noexcept
  {
    single_result = result;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


template<class Executor>
void test_then_receiver_of_int_and_return_void(Executor ex)
{
  auto f1 = make_ready_host_future(ex, 13);

  int expected = 13;
  single_result = -1;
  auto f2 = std::move(f1).then(ex, receive_int_and_return_void{expected});

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


template<class Executor>
void test_then_receiver_of_int_and_return_int(Executor ex)
{
  int expected = 13;
  auto f1 = make_ready_host_future(ex, expected - 1);

  auto f2 = std::move(f1).then(ex, receive_and_add_one{});

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


template<class Executor>
void test_bulk_then_receiver_of_int(Executor ex)
{
  int expected = 7;
  auto f1 = make_ready_host_future(ex, std::move(expected));

  bulk_result0 = false;
  bulk_result1 = false;

  auto f2 = std::move(f1).bulk_then(ex, many_receiver_of_int{expected}, 2);

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


template<class Executor>
void test_bulk_then_receiver_of_void(Executor ex)
{
  auto f1 = make_ready_host_future(ex);

  bulk_result0 = false;
  bulk_result1 = false;

  auto f2 = std::move(f1).bulk_then(ex, many_receiver_of_void{}, 2);

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(bulk_result0);
  assert(bulk_result1);
}


template<class Executor>
void test_bulk_then_void_to_void(Executor ex)
{
  auto f1 = make_ready_host_future(ex);

  bulk_result0 = false;
  bulk_result1 = false;

  auto f2 = std::move(f1).bulk_then(ex, [] __host__ __device__ (std::size_t idx)
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


template<class Executor>
void test_bulk_then_int_to_int(Executor ex)
{
  int expected = 13;
  auto f1 = make_ready_host_future(ex, std::move(expected));

  bulk_result0 = false;
  bulk_result1 = false;

  auto f2 = std::move(f1).bulk_then(ex, [] __host__ __device__ (std::size_t idx, int& value)
  {
    switch(idx)
    {
      case 0:
      {
        bulk_result0 = (13 == value);
        break;
      }

      case 1:
      {
        bulk_result1 = (13 == value);
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


template<class Executor>
void test_bulk(Executor ex)
{
  {
    // test void future

    bulk_result0 = false;
    bulk_result1 = false;

    ns::submit(make_ready_host_future(ex).bulk(2), many_receiver_of_void{});
    assert(cudaSuccess == cudaDeviceSynchronize());

    assert(bulk_result0);
    assert(bulk_result1);
  }

  {
    // test int future

    bulk_result0 = false;
    bulk_result1 = false;

    ns::submit(make_ready_host_future(ex, 13).bulk(2), many_receiver_of_int{13});
    assert(cudaSuccess == cudaDeviceSynchronize());

    assert(bulk_result0);
    assert(bulk_result1);
  }
}


template<class StreamExecutor>
void test(StreamExecutor ex)
{
  test_move_construction(ex);
  test_move_assignment(ex);
  test_get(ex);

  test_then_void_to_void(ex);
  test_then_void_to_int(ex);
  test_then_int_to_void(ex);
  test_then_int_to_int(ex);
  test_then_int_to_float(ex);

  test_then_receiver_of_void_and_return_void(ex);
  test_then_receiver_of_void_and_return_int(ex);
  test_then_receiver_of_int_and_return_void(ex);
  test_then_receiver_of_int_and_return_int(ex);

  test_bulk_then_receiver_of_void(ex);
  test_bulk_then_receiver_of_int(ex);

  test_bulk_then_void_to_void(ex);
  test_bulk_then_int_to_int(ex);

  test_bulk(ex);
}


void test_host_future()
{
#ifdef __CUDACC__
  // stream_executor launches kernels, so only test it with a CUDA compiler
  test(ns::execution::stream_executor{});
#endif

  test(ns::execution::callback_executor{});
}

