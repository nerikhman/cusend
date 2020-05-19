This directory collects functionality related to [futures and promises](https://en.wikipedia.org/wiki/Futures_and_promises).

Futures are objects that represent the eventual completion of a computation at some point in the *future*.

Promises are objects that represents an intention or *promise* to execute a computation.

There are two class templates:

  1. `future<T>`
  2. `host_promise<T>`

`future<T>` represents the eventual completion of a computation executing on one or many devices. It presents an interface similar to [`std::future<T>`](https://en.cppreference.com/w/cpp/thread/future).

`host_promise<T>` represents a promise to execute a computation on a CPU and deliver the result of that computation to a device. It presents an interface similar to [`std::promise<T>`](https://en.cppreference.com/w/cpp/thread/promise).

Importantly, `host_promise::get_future()` retrieves a special type of future representing the eventual completion of a CPU computation that may be used to create dependent work on one or many devices.

