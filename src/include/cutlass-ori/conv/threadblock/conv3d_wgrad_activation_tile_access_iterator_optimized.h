/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Templates implementing loading of convolution tiles mapped to GEMM B (activation tile) 
    matrix from memory.

    This iterator assumes TensorNDHWC layout of tensors in Global Memory.

    The iterator is specialized for each of the three convolution operators: forward propagation (Fprop),
    backward data gradient (Dgrad), and backward weight gradient (Wgrad).
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv3d_problem_size.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Shape_,
  typename Element_,
  typename ThreadMap_
>
class Conv3dWgradActivationTileAccessIteratorOptimized {
public:

  //
  // Types
  //
  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::TensorNDHWC;
  using ThreadMap = ThreadMap_;
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kOptimized;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 3;
  using ConvProblemSize = typename conv::Conv3dProblemSize;
  
  static_assert(sizeof_bits<Element>::value >= 8,
    "WGRAD requires elements of size 8b or greater.");

  //
  // Parameters structure
  //

  struct Params {

    Layout layout;

    int RSC;                  // product of R*S*C
    unsigned rsc_mul;         // precomputed quantities for fast computation of div/% by RSC
    unsigned rsc_shr;         //    in device code.

    int SC;                   // product of S*C
    unsigned sc_mul;          // precomputed quantities for fast computation of div/% by SC
    unsigned sc_shr;          //    in device code.

    unsigned c_mul;          // precomputed quantities for fast computation of div/% by C
    unsigned c_shr;          //    in device code.

    int ZPQ;                 // product of Z*P*Q
    unsigned zpq_mul;        // precomputed quantities for fast computation of div/% by ZPQ
    unsigned zpq_shr;        //    in device code.

    int PQ;                  // product of P*Q
    unsigned pq_mul;         // precomputed quantities for fast computation of div/% by PQ
    unsigned pq_shr;         //    in device code.

    unsigned q_mul;          // precomputed quantities for fast computation of div/% by Q
    unsigned q_shr;          //    in device code.

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(
      Conv3dProblemSize const &problem_size, 
      Layout const &layout
    ): layout(layout) {

      // Precompute several quantities for fast modulo arithmetic.
      RSC = problem_size.R * problem_size.S * problem_size.C;
      find_divisor(rsc_mul, rsc_shr, RSC);

      SC = problem_size.S * problem_size.C;
      find_divisor(sc_mul, sc_shr, SC);
      
      find_divisor(c_mul, c_shr, problem_size.C);

      ZPQ = problem_size.Z * problem_size.P * problem_size.Q;
      find_divisor(zpq_mul, zpq_shr, ZPQ);

      PQ = problem_size.P * problem_size.Q;
      find_divisor(pq_mul, pq_shr, PQ);

      find_divisor(q_mul, q_shr, problem_size.Q);

    }
  };

private:

  Params const &params_;
  Conv3dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  char const *pointer_;

  // Precomputed effective filter postion (t,r,s) in contiguous dimension stays constant for each gemm_iteration_k
  // required for nzpq -> ndhw translation
  int precomputed_filter_t_[ThreadMap::Iterations::kContiguous];
  int precomputed_filter_r_[ThreadMap::Iterations::kContiguous];
  int precomputed_filter_s_[ThreadMap::Iterations::kContiguous];

  // Channel dimension in contiguous dimension stays constant for each gemm_iteration_k
  int filter_c_[ThreadMap::Iterations::kContiguous];

  int offset_nzpq_[ThreadMap::Iterations::kStrided];

public:

  CUTLASS_HOST_DEVICE
  Conv3dWgradActivationTileAccessIteratorOptimized(
    Params const &params, 
    Conv3dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params), 
    problem_size_(problem_size), 
    pointer_(reinterpret_cast<char const *>(ptr)) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);
    
    // initialize t,r,s,c filter position for every contiguous iteration
    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

      int trsc_offset = threadblock_offset.column() + thread_coord.contiguous()
                        + c * ThreadMap::Delta::kContiguous;

      // The subseqnet fast_divmod() operations are equivalent to the following logical computation:
      //
      // 
      // filter_t_[c] = trsc_offset / (problem_size_.R * problem_size_.S * problem_size_.C);
      // int residual = trsc_offset % (problem_size_.R * problem_size_.S * problem_size_.C);
      //
      // filter_r_[c] = residual / (problem_size_.S * problem_size_.C);
      // residual = residual % (problem_size_.S * problem_size_.C);
      //
      // filter_s_[c] = residual / problem_size_.C;
      // filter_c_[c] = residual % problem_size_.C;

      int residual;
      fast_divmod(precomputed_filter_t_[c], residual, trsc_offset, params_.RSC, params_.rsc_mul, params_.rsc_shr);
      fast_divmod(precomputed_filter_r_[c], residual, residual, params_.SC, params_.sc_mul, params_.sc_shr);
      fast_divmod(precomputed_filter_s_[c], filter_c_[c], residual, problem_size_.C, params_.c_mul, params_.c_shr);

      int t = precomputed_filter_t_[c];
      int r = precomputed_filter_r_[c];
      int s = precomputed_filter_s_[c];

      if (problem_size_.mode == Mode::kConvolution) {
        t = (problem_size_.T - 1 - t);
        r = (problem_size_.R - 1 - r);
        s = (problem_size_.S - 1 - s);
      }
      
      // efective t,r,s for every contiguous dimension
      precomputed_filter_t_[c] = - problem_size_.pad_d + t * problem_size_.dilation_d;
      precomputed_filter_r_[c] = - problem_size_.pad_h + r * problem_size_.dilation_h;
      precomputed_filter_s_[c] = - problem_size_.pad_w + s * problem_size_.dilation_w;


    }

    // initialize n, z, p, q offset for every strided iteration
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
    
      offset_nzpq_[s] = threadblock_offset.row() + thread_coord.strided() 
                      + s * ThreadMap::Delta::kStrided;   
    }
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(Index index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  CUTLASS_HOST_DEVICE
  void advance() {
    
    // moves to the next GEMM-K offset (offset_nzpq_) in GEMM-B by a CTA-K tile
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      offset_nzpq_[s] += Shape::kRow * problem_size_.split_k_slices;
    }
  }

  /// Returns the coordinate in the activation tensor x that is currently pointed to
  /// by the iterator.

  CUTLASS_HOST_DEVICE
  TensorCoord at() const {

    // The subseqnet fast_divmod() operations are equivalent to the following logical computation:
    //
    //
    // int n = offset_nzpq_[iteration_strided_] / (problem_size_.Z * problem_size_.P * problem_size_.Q);
    // int residual = offset_nzpq_[iteration_strided_] % (problem_size_.Z * problem_size_.P * problem_size_.Q);
    // 
    // int z = residual / (problem_size_.P * problem_size_.Q);
    // residual = residual % (problem_size_.P * problem_size_.Q);
    // 
    // int p = residual / problem_size_.Q;
    // int q = residual % problem_size_.Q;

    int residual, n, z, p, q;
    fast_divmod(n, residual, offset_nzpq_[iteration_strided_], params_.ZPQ, params_.zpq_mul, params_.zpq_shr);
    fast_divmod(z, residual, residual, params_.PQ, params_.pq_mul, params_.pq_shr);
    fast_divmod(p, q, residual, problem_size_.Q, params_.q_mul, params_.q_shr);
 
    int d = z * problem_size_.stride_d + precomputed_filter_t_[iteration_contiguous_];
    int h = p * problem_size_.stride_h + precomputed_filter_r_[iteration_contiguous_];;
    int w = q * problem_size_.stride_w + precomputed_filter_s_[iteration_contiguous_];

    return TensorCoord(n, d, h, w, filter_c_[iteration_contiguous_]);
  }

  /// Returns true if the current coordinate is within the activation tensor x
  CUTLASS_HOST_DEVICE
  bool valid() const {
    TensorCoord coord = at();

    return coord.n() < problem_size_.N &&
      coord.d() >= 0 && coord.d() < problem_size_.D &&
      coord.h() >= 0 && coord.h() < problem_size_.H &&
      coord.w() >= 0 && coord.w() < problem_size_.W &&
      coord.c() < problem_size_.C;
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_DEVICE
  AccessType const *get() const {

    TensorCoord coord = at();
    LongIndex offset = params_.layout(coord);

    return reinterpret_cast<AccessType const *>(pointer_ + offset * sizeof_bits<Element>::value / 8);
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv3dWgradActivationTileAccessIteratorOptimized &operator++() {
    ++iteration_contiguous_;
    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
      return *this;
    }
    iteration_contiguous_ = 0;
    ++iteration_strided_;
    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }
    iteration_strided_ = 0;
 
    return *this;
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv3dProblemSize const &problem_size) {

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.K % (128/sizeof_bits<Element>::value)) {
      return Status::kErrorInvalidProblem;
    }

    return Status::kSuccess;
  }
  
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////


