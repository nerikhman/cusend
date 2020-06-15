#include <cassert>
#include <cusend/sender/sender_traits.hpp>
#include <cusend/sender/set_value.hpp>


struct my_sender
{
  template<class R>
  void submit(R&& r) &&
  {
    cusend::set_value(r);
  }
};


struct not_a_sender {};


struct my_sender_with_value_types : my_sender
{
  template<template<class...> class Tuple, template<class...> class Variant>
  using value_types = Variant<Tuple<>>;
};

struct my_sender_with_value_types_and_error_types : my_sender
{
  template<template<class...> class Tuple, template<class...> class Variant>
  using value_types = Variant<Tuple<>>;

  template<template<class...> class Variant>
  using error_types = Variant<>;
};


struct my_sender_with_traits : my_sender
{
  template<template<class...> class Tuple, template<class...> class Variant>
  using value_types = Variant<Tuple<>>;

  template<template<class...> class Variant>
  using error_types = Variant<>;

  constexpr static bool sends_done = false;
};


struct my_executor
{
  template<class F>
  void execute(F&& f) const;

  bool operator==(const my_executor&) const;
  bool operator!=(const my_executor&) const;
};


template<class...>
struct variadic;


void test_sender_traits()
{
  {
    // test a sender without any traits
    using traits = cusend::sender_traits<my_sender>;

    static_assert(!cusend::detail::has_sender_types<traits>::value, "Expected sender not to have types.");
  }

  {
    // test a sender with a single trait
    using traits = cusend::sender_traits<my_sender_with_value_types>;

    static_assert(!cusend::detail::has_sender_types<traits>::value, "Expected sender not to have types.");
  }

  {
    // test a sender with two traits
    using traits = cusend::sender_traits<my_sender_with_value_types_and_error_types>;

    static_assert(!cusend::detail::has_sender_types<traits>::value, "Expected sender not to have types.");
  }

  {
    // test an executor
    using traits = cusend::sender_traits<my_executor>;

    using value_types = traits::value_types<variadic,variadic>;
    using error_types = traits::error_types<variadic>;

    static_assert(std::is_same<value_types, variadic<variadic<>>>::value, "Expected something else.");
    static_assert(std::is_same<error_types, variadic<std::exception_ptr>>::value, "Expected something else.");
    static_assert(traits::sends_done, "Expected sends_done == true.");
  }

  {
    // test a sender with all traits
    using traits = cusend::sender_traits<my_sender_with_traits>;

    using value_types = traits::value_types<variadic,variadic>;
    using error_types = traits::error_types<variadic>;

    static_assert(std::is_same<value_types, variadic<variadic<>>>::value, "Expected something else.");
    static_assert(std::is_same<error_types, variadic<>>::value, "Expected something else.");
    static_assert(!traits::sends_done, "Expected sends_done == false.");
  }

  {
    // test a reference to a sender with all traits
    using traits = cusend::sender_traits<my_sender_with_traits&>;

    using value_types = traits::value_types<variadic,variadic>;
    using error_types = traits::error_types<variadic>;

    static_assert(std::is_same<value_types, variadic<variadic<>>>::value, "Expected something else.");
    static_assert(std::is_same<error_types, variadic<>>::value, "Expected something else.");
    static_assert(!traits::sends_done, "Expected sends_done == false.");
  }
}

