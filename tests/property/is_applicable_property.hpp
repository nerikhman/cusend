#include <type_traits>
#include <cusend/property/is_applicable_property.hpp>

namespace ns = cusend;

struct has_is_applicable_property_function
{
  template<class T>
  constexpr static bool is_applicable_property()
  {
    return std::is_integral<T>::value;
  }
};

struct has_is_applicable_property_variable
{
  template<class T>
  constexpr static bool is_applicable_property_v = std::is_integral<T>::value;
};


void test_is_applicable_property()
{
  static_assert(ns::is_applicable_property<int, has_is_applicable_property_function>::value, "Error.");
  static_assert(ns::is_applicable_property_v<int, has_is_applicable_property_function>, "Error.");

  static_assert(!ns::is_applicable_property<float, has_is_applicable_property_function>::value, "Error.");
  static_assert(!ns::is_applicable_property_v<float, has_is_applicable_property_function>, "Error.");
}

