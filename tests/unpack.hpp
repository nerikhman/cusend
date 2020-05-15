#include <cassert>
#include <cstring>
#include <cusend/just.hpp>
#include <cusend/unpack.hpp>
#include <exception>
#include <utility>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __global__
#endif


template<class Tuple>
struct tuple_receiver
{
  Tuple& result;

  template<class... Args>
  __host__ __device__
  void set_value(Args&&... args) &&
  {
    result = Tuple{std::forward<Args>(args)...};
  }

  __host__ __device__
  void set_error(std::exception_ptr) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


template<class Tuple>
struct my_sender_with_member_function
{
  Tuple result;

  __host__ __device__
  auto unpack() &&
    -> decltype(cusend::unpack(cusend::just(result)))
  {
    return cusend::unpack(cusend::just(result));
  }

  template<class R>
  __host__ __device__
  auto connect(R&& r) &&
    -> decltype(cusend::connect(std::forward<R>(r)))
  {
    return cusend::connect(std::forward<R>(r));
  }
};


template<class Tuple>
struct my_sender_with_free_function
{
  Tuple result;
};


template<class Tuple>
__host__ __device__
auto unpack(my_sender_with_free_function<Tuple>&& s)
  -> decltype(cusend::unpack(cusend::just(s.result)))
{
  return cusend::unpack(cusend::just(s.result));
}


template<class S, class Tuple>
__host__ __device__
void test(S tuple_sender, Tuple expected)
{
  Tuple result;

  auto sender = cusend::unpack(std::move(tuple_sender));

  static_assert(cusend::is_typed_sender<decltype(sender)>::value, "Error.");

  std::move(sender).connect(tuple_receiver<Tuple>{result}).start();

  assert(expected == result);
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

void test_unpack()
{
  using namespace cusend;

  test(just(detail::make_tuple()), detail::make_tuple());
  test(just(detail::make_tuple(13)), detail::make_tuple(13));
  test(just(detail::make_tuple(13,7)), detail::make_tuple(13,7));
  test(just(detail::make_tuple(13,7,42)), detail::make_tuple(13,7,42));

  // test with some customized senders
  auto tuple = detail::make_tuple(13,7,42);
  test(my_sender_with_member_function<decltype(tuple)>{tuple}, tuple);
  test(my_sender_with_free_function<decltype(tuple)>{tuple}, tuple);

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] __device__ ()
  {
    test(just(detail::make_tuple()), detail::make_tuple());
    test(just(detail::make_tuple(13)), detail::make_tuple(13));
    test(just(detail::make_tuple(13,7)), detail::make_tuple(13,7));
    test(just(detail::make_tuple(13,7,42)), detail::make_tuple(13,7,42));

    // test with some customized senders
    auto tuple = detail::make_tuple(13,7,42);
    test(my_sender_with_member_function<decltype(tuple)>{tuple}, tuple);
    test(my_sender_with_free_function<decltype(tuple)>{tuple}, tuple);
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

