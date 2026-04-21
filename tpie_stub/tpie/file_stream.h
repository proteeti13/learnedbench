#pragma once
#include <stdexcept>
// Stub: TPIE not installed. These types compile but must never be called at runtime.
namespace tpie {
    template<typename T>
    class file_stream {
    public:
        void open(const std::string&) { throw std::runtime_error("TPIE stub: open() not implemented"); }
        void close() {}
        T read() { throw std::runtime_error("TPIE stub: read() not implemented"); return T{}; }
        void write(const T&) { throw std::runtime_error("TPIE stub: write() not implemented"); }
    };
}
