This directory contains invocables which are used in the implementation of `future::then` and `detail::host_future::then`.

Most of them adapt a receiver into an invocable which calls `set_value` and does something with its result.

