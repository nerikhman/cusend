#include <cassert>
#include <exception>
#include <cudex/sender/is_sender_to.hpp>
#include <cudex/sender/sender_base.hpp>


struct my_receiver
{
  void set_value() &&;
  void set_done() && noexcept;
  void set_error(std::exception_ptr) && noexcept;
};


struct not_a_receiver {};


struct sender_to_my_receiver : cudex::sender_base
{
  void connect(my_receiver&&) &&;
};


struct not_a_sender {};


void test_is_sender_to()
{
  using namespace cudex;

  // test not a sender
  static_assert(!is_sender_to<not_a_sender, my_receiver>::value, "Expected false.");

  // test sender to my_receiver
  static_assert(is_sender_to<sender_to_my_receiver, my_receiver>::value, "Expected true.");

  // test a reference to my_sender to my_receiver
  static_assert(is_sender_to<sender_to_my_receiver&&, my_receiver>::value, "Expected true.");

  // test sender to not_a_receiver
  static_assert(!is_sender_to<sender_to_my_receiver, not_a_receiver>::value, "Expected false.");

  // test a reference to not_a_receiver
  static_assert(!is_sender_to<sender_to_my_receiver&&, not_a_receiver>::value, "Expected true.");
}

