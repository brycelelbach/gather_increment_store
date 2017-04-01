// Copyright (c) 2017 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// g++ -Wall -std=c++1z -O3 -fno-rtti -fno-exceptions -ftree-vectorize -fvect-cost-model=unlimited -save-temps -march=knl -mavx512f -mavx512pf -mavx512cd -S autovectorized_gather_increment_store.cpp -o autovectorized_gather_increment_store.gcc.asm

#include "vectorization_and_assumption_hints.hpp"

namespace std { using ptrdiff_t = decltype("me" - "ow"); } 

void scalar_gather_increment_store(
    double const* __restrict__   src
  , double* __restrict__         dest
  , std::ptrdiff_t* __restrict__ dest_indices
  , std::ptrdiff_t               N
    ) noexcept
{
    BOOST_ASSUME_ALIGNED(src,          64);
    BOOST_ASSUME_ALIGNED(dest,         64);
    BOOST_ASSUME_ALIGNED(dest_indices, 64);

    BOOST_ASSUME((N % 64) == 0);

    for (int k = 0; k != N; ++k)
        dest[dest_indices[k]] += src[k];
}

