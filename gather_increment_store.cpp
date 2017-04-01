// Copyright (c) 2017 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// g++ -Wall -std=c++1z -O3 -save-temps -march=knl -mavx512f -mavx512pf -mavx512cd gather_increment_store.cpp -o gather_increment_store 

#include <immintrin.h>
#include <array>
#include <algorithm>
#include <limits>
#include <iostream>

template <typename T>
constexpr bool fp_equals(T x, T y, T epsilon = std::numeric_limits<T>::epsilon()) noexcept
{
    static_assert(std::is_floating_point<T>::value, "T must be a floating point type.");
    return (((x + epsilon >= y) && (x - epsilon <= y)) ? true : false);
}

template <std::size_t N>
__m512d vector_gather_increment_store(__m512d src, std::array<double, N>& dest, __m512i dest_indices) noexcept
{
    __m512d const gathered_dest = _mm512_i64gather_pd(dest_indices, dest.data(), sizeof(double));

    __m512d const dest_plus_src = _mm512_add_pd(gathered_dest, src);

    _mm512_i64scatter_pd(dest.data(), dest_indices, dest_plus_src, sizeof(double));

    return gathered_dest;
}

template <std::size_t N>
void scalar_gather_increment_store(__m512d src, std::array<double, N>& dest, __m512i dest_indices) noexcept
{
    for (int k = 0; k != 8; ++k)
        dest[dest_indices[k]] += src[k];
}

template <std::size_t N>
bool verify_gather_increment_store(__m512d src, std::array<double, N>& dest, __m512i dest_indices) noexcept
{
    std::array<double, N> expected_dest;
    std::copy(dest.begin(), dest.end(), expected_dest.begin());

    scalar_gather_increment_store(src, expected_dest, dest_indices);

    __m512d const gathered_dest = vector_gather_increment_store(src, dest, dest_indices);

    std::cout << "     dest ["; for (int k = 0; k != 8; ++k) std::cout << gathered_dest[k]               << " "; std::cout << "]\n";
    std::cout << " +=   src ["; for (int k = 0; k != 8; ++k) std::cout << src[k]                         << " "; std::cout << "]\n"; 
    std::cout << "--------------------------------------------------------------------------------------\n";
    std::cout << " observed ["; for (int k = 0; k != 8; ++k) std::cout << dest[dest_indices[k]]          << " "; std::cout << "]\n"; 
    std::cout << " expected ["; for (int k = 0; k != 8; ++k) std::cout << expected_dest[dest_indices[k]] << " "; std::cout << "]\n"; 

    bool passed = true;
    for (int k = 0; k != 8; ++k)
        passed = passed && fp_equals(dest[dest_indices[k]], expected_dest[dest_indices[k]]);

    std::cout << (passed ? "PASSED." : "FAILED.") << "\n";

    return passed;
}

int main()
{
    {
        __m512d const src = _mm512_set1_pd(1.0);

        std::array<double, 8> dest;
        std::fill(dest.begin(), dest.end(), 0.0);

        __m512i const dest_indices = { 0, 1, 2, 3, 4, 5, 6, 7 };

        verify_gather_increment_store(src, dest, dest_indices);
    }

    std::cout << "\n";

    {
        __m512d const src = _mm512_set1_pd(1.0);

        std::array<double, 8> dest;
        std::fill(dest.begin(), dest.end(), 0.0);

        __m512i const dest_indices = { 0, 0, 0, 0, 1, 1, 1, 1 };

        verify_gather_increment_store(src, dest, dest_indices); // FAILS: Conflicts.
    }

    std::cout << "\n";

    {
        __m512d const src = _mm512_set1_pd(1.0);

        std::array<double, 8> dest;
        std::fill(dest.begin(), dest.end(), 0.0);

        __m512i const dest_indices = { 0, 1, 0, 1, 0, 1, 0, 1 };

        verify_gather_increment_store(src, dest, dest_indices); // FAILS: Conflicts.
    }
}

