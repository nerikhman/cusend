#include <cassert>
#include <exception>
#include <cusend/sender/is_receiver.hpp>


struct my_receiver
{
  void set_value() &&;

  void set_error(std::exception_ptr) &&;

  void set_done() &&;
};


struct not_a_receiver {};


void test_is_receiver()
{
  {
    // test not a receiver
    static_assert(!cusend::is_receiver<not_a_receiver, std::exception_ptr>::value, "Expected not a receiver.");
  }

  {
    // test a receiver of exception_ptr
    static_assert(cusend::is_receiver<my_receiver, std::exception_ptr>::value, "Expected a receiver.");
  }

  {
    // test a receiver of void
    static_assert(!cusend::is_receiver<my_receiver, void>::value, "Expected not a receiver.");
  }

  {
    // test a reference to a receiver of exception_ptr
    static_assert(cusend::is_receiver<my_receiver&&, std::exception_ptr>::value, "Expected a receiver.");
  }
}

