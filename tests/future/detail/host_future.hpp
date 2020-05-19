#include <cassert>
#include <utility>
#include <cusend/future/host_promise.hpp>

namespace ns = cusend;

void test_move_construction()
{
  {
    ns::host_promise<int> p0{};
    ns::detail::host_future<int> f0 = p0.get_future();
    assert(f0.valid());

    ns::detail::host_future<int> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }

  {
    ns::host_promise<void> p0;
    ns::detail::host_future<void> f0 = p0.get_future();
    assert(f0.valid());

    ns::detail::host_future<void> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }
}


void test_move_assignment()
{
  {
    ns::host_promise<int> p0{};
    ns::detail::host_future<int> f0 = p0.get_future();
    assert(f0.valid());

    ns::detail::host_future<int> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }


  {
    ns::host_promise<void> p0{};
    ns::detail::host_future<void> f0 = p0.get_future();
    assert(f0.valid());

    ns::detail::host_future<void> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }
}


template<class T>
ns::detail::host_future<T> make_ready_host_future(T value)
{
  ns::host_promise<T> p{};
  std::move(p).set_value(value);
  return p.get_future();
}


ns::detail::host_future<void> make_ready_host_future()
{
  ns::host_promise<void> p{};
  std::move(p).set_value();
  return p.get_future();
}


void test_get()
{
  {
    ns::detail::host_future<int> f = make_ready_host_future(13);
    assert(f.valid());

    assert(13 == std::move(f).get());
    assert(!f.valid());
  }

  {
    ns::detail::host_future<void> f = make_ready_host_future();
    assert(f.valid());

    std::move(f).get();
    assert(!f.valid());
  }
}


// future.then has several different code paths
// that depend on the type of input and output future
// these tests exercise each case of .then


void test_then_void_to_void()
{
  // then void -> void
  auto f1 = make_ready_host_future();
  assert(f1.valid());

  auto f2 = std::move(f1).then([]{ return; });

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  std::move(f2).get();
  assert(!f2.valid());
}


void test_then_int_to_void()
{
  // then int -> void
  auto f1 = make_ready_host_future(7);
  assert(f1.valid());

  auto f2 = std::move(f1).then([](int){return;});

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
  auto f1 = make_ready_host_future();

  auto f2 = std::move(f1).then([]{return 13;});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(std::move(f2).get() == 13);
  assert(!f2.valid());
}


void test_then_int_to_int()
{
  // then int -> int
  auto f1 = make_ready_host_future(7);

  auto f2 = std::move(f1).then([] __host__ __device__ (int arg) { return arg + 6; });

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
  auto f1 = make_ready_host_future(7);

  auto f2 = std::move(f1).then([](int arg){return static_cast<float>(arg + 6);});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(std::move(f2).get() == 13.f);
  assert(!f2.valid());
}


struct receive_at
{
  int* address;

  void set_value() && noexcept
  {
    *address = true;
  }

  void set_value(int value) && noexcept
  {
    std::cout << "receive_at::set_value: received " << value << std::endl;
    *address = value;
  }

  template<class E>
  void set_error(E&&) && noexcept {}

  void set_done() && noexcept {}
};


void test_then_receiver()
{
  {
    // receive int

    int expected = 7;
    auto f1 = make_ready_host_future(expected);

    int result = -1;
    auto f2 = std::move(f1).then(receive_at{&result});

    assert(!f1.valid());
    assert(f2.valid());
    f2.wait();

    assert(f2.is_ready());
    assert(expected == result);
  }

  {
    // receive void
    auto f1 = make_ready_host_future();

    int result = false;
    auto f2 = std::move(f1).then(receive_at{&result});

    assert(!f1.valid());
    assert(f2.valid());
    f2.wait();

    assert(f2.is_ready());
    assert(result);
  }
}


void test_host_future()
{
  test_move_construction();
  test_move_assignment();
  test_get();
  test_then_void_to_void();
  test_then_int_to_void();
  test_then_void_to_int();
  test_then_int_to_int();
  test_then_int_to_float();
  test_then_receiver();
}

