#include <cassert>
#include <cusend/future/host_promise.hpp>

namespace ns = cusend;

struct move_only
{
  int value;

  move_only(int v) : value{v} {}

  move_only(move_only&&) = default;
};


void test_default_construction()
{
  {
    ns::host_promise<int> p{};
    auto f = p.get_future();
    std::move(p).set_value(13);
    assert(13 == f.get());
  }

  {
    ns::host_promise<void> p{};
    auto f = p.get_future();
    std::move(p).set_value();
    f.get();
  }
}


void test_move_construction()
{
  {
    ns::host_promise<int> p0{};
    ns::host_promise<int> p1 = std::move(p0);
    auto f = p1.get_future();
    std::move(p1).set_value(13);
    assert(13 == f.get());
  }

  {
    ns::host_promise<void> p0{};
    ns::host_promise<void> p1 = std::move(p0);
    auto f = p1.get_future();
    std::move(p1).set_value();
    f.get();
  }
}


void test_set_value()
{
  {
    // move-only value
    ns::host_promise<move_only> p{};
    auto f = p.get_future();
    std::move(p).set_value(move_only{13});
    assert(13 == f.get().value);
  }

  {
    // copyable value
    ns::host_promise<int> p{};
    auto f = p.get_future();
    int expected = 13;
    std::move(p).set_value(expected);
    assert(expected == f.get());
  }

  {
    // void value
    ns::host_promise<void> p{};
    auto f = p.get_future();
    std::move(p).set_value();
    f.get();
  }
}


void test_set_error()
{
  {
    // non-void value

    ns::host_promise<int> p{};
    auto f = p.get_future();
    std::move(p).set_error(std::make_exception_ptr(13));

    try
    {
      f.get();
      assert(false);
    }
    catch(int e)
    {
      assert(13 == e);
    }
    catch(...)
    {
      assert(false);
    }
  }

  {
    // void value

    ns::host_promise<void> p{};
    auto f = p.get_future();
    std::move(p).set_error(std::make_exception_ptr(13));

    try
    {
      f.get();
      assert(false);
    }
    catch(int e)
    {
      assert(13 == e);
    }
    catch(...)
    {
      assert(false);
    }
  }
}


void test_set_done()
{
  {
    // non-void value

    ns::host_promise<int> p{};
    auto f = p.get_future();
    std::move(p).set_done();

    try
    {
      f.get();
      assert(false);
    }
    catch(std::future_error e)
    {
      assert(std::future_errc::broken_promise == e.code());
    }
    catch(...)
    {
      assert(false);
    }
  }

  {
    // void value

    ns::host_promise<void> p{};
    auto f = p.get_future();
    std::move(p).set_done();

    try
    {
      f.get();
      assert(false);
    }
    catch(std::future_error e)
    {
      assert(std::future_errc::broken_promise == e.code());
    }
    catch(...)
    {
      assert(false);
    }
  }
}


void test_host_promise()
{
  test_default_construction();
  test_move_construction();
  test_set_value();
  test_set_error();
  test_set_done();
}

