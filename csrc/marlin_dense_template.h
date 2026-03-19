/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

// Dependency aggregator for the dense Marlin<> GEMM kernel.
// Provides marlin.cuh, marlin_dtypes.cuh, dequant.h and marlin_mma.h at
// global scope.  marlin_dense.cu then opens a single namespace marlin_dense {}
// and includes marlin_unified.h inside that block.

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_dense
#endif

#include "quantization/marlin/marlin.cuh"
#include "quantization/marlin/marlin_dtypes.cuh"
#include "quantization/marlin/dequant.h"
#include "quantization/marlin/marlin_mma.h"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");
