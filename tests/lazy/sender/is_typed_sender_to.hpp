#include <cassert>
#include <cusend/lazy/sender/is_typed_sender_to.hpp>
#include <exception>


struct not_a_receiver {};

struct my_receiver
{
  void set_value() &&;
  void set_value(int) &&;
  void set_value(int, float) &&;
  void set_value(float, float, float) &&;
  template<class... Args>
  void set_value(int, std::string, Args&&... args) &&;

  void set_done() noexcept;
  void set_exception(std::exception_ptr) noexcept;
};


struct not_a_sender {};

struct untyped_sender_to_my_receiver : cusend::sender_base
{
  void connect(my_receiver&&);
};

struct typed_sender_to_my_receiver : untyped_sender_to_my_receiver
{
  template<template<class...> class Tuple, template<class...> class Variant>
  using value_types = Variant<
    Tuple<int>,
    Tuple<int,float>,
    Tuple<int, std::string, std::string>,
    Tuple<float, float, float>
  >;

  template<template<class...> class Variant>
  using error_types = Variant<>;

  constexpr static bool sends_done = false;
};


void test_is_typed_sender_to()
{
  using namespace cusend;

  // not_a_sender, not_a_receiver
  static_assert(!is_typed_sender_to<not_a_sender, not_a_receiver>::value, "Expected false.");

  // not_a_sender, my_receiver
  static_assert(!is_typed_sender_to<not_a_sender, my_receiver>::value, "Expected false.");

  // untyped_sender_to_my_receiver, not_a_receiver 
  static_assert(!is_typed_sender_to<untyped_sender_to_my_receiver, not_a_receiver>::value, "Expected false.");

  // untyped_sender_to_my_receiver, my_receiver 
  static_assert(!is_typed_sender_to<untyped_sender_to_my_receiver, my_receiver>::value, "Expected false.");

  // typed_sender_to_my_receiver, not_a_receiver
  static_assert(!is_typed_sender_to<typed_sender_to_my_receiver, not_a_receiver>::value, "Expected false.");

  // typed_sender_to_my_receiver, my_receiver
  static_assert(is_typed_sender_to<typed_sender_to_my_receiver, my_receiver>::value, "Expected true.");

  // typed_sender_to_my_receiver&, my_receiver
  static_assert(is_typed_sender_to<typed_sender_to_my_receiver&, my_receiver>::value, "Expected true.");

  // typed_sender_to_my_receiver&&, my_receiver
  static_assert(is_typed_sender_to<typed_sender_to_my_receiver&&, my_receiver>::value, "Expected true.");
}

