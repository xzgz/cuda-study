
struct DefaultMma<float, LayoutA, kAlignmentA, float, LayoutB,
                  kAlignmentB, float, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false, SharedMemoryClearOption::kNone,
                  GatherA, GatherB> {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, float, LayoutA, float,
      LayoutB, float, layout::RowMajor, arch::OpClassTensorOp, 2,
      arch::OpMultiplyAddFastF16>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          float, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA, GatherA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          float, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB, GatherB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, float,
      layout::RowMajor, typename MmaCore::MmaPolicy>;
};

template<typename WarpShape>
constexpr int simt_get_warp_threads_m() {
    return (WarpShape::kM > WarpShape::kN) ? 8 : 4;
}


cutlass::gemm::threadblock::DefaultMmaCore:

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);

  static_assert(!(kPaddingM % LaneM),
                "Padding must be divisible by Lane");

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
      WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,     /// Data type of A elements
      SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
      ElementB,     /// Data type of B elements
      SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
      ElementC,     /// Element type of C matrix
      LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
      Policy        /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
  >;

  /// Policy used to define MmaPipelined 
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<kPaddingM, 0>,    // skew for A matrix to avoid SMEM bank conflicts
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
};


cutlass::gemm::warp::MmaSimt:

  /// Iterates over the A operand in memory
  using IteratorA = MmaSimtTileIterator<
    MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>,
    Operand::kA,
    ElementA,
    LayoutA,
    Policy,
    PartitionsK,
    Shape::kK
  >;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA = FragmentA;

  /// Iterates over the B operand in memory
  using IteratorB = MmaSimtTileIterator<
    MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>,
    Operand::kB,
    ElementB,
    LayoutB,
    Policy,
    PartitionsK,
    Shape::kK
  >;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;


MmaSimtTileIterator:

/// Operand tag
  static Operand const kOperand = Operand::kB;

  static_assert(!(Shape::kColumn % Policy::WarpShape::kColumn), 
    "The warp-level GEMM N size must be divisible by the number of threads arranged along the N dimension.");
  
  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
  static_assert(Policy::WarpShape::kColumn > 0, "Policy::WarpShape::kColumn must be greater than zero.");
  static_assert(Shape::kColumn / Policy::WarpShape::kColumn > 0, "Shape::kColumn / Policy::WarpShape::kColumn must be greater than zero.");

  /// Thread-level shape of a fragment
  using ThreadShape = MatrixShape<
    Shape::kRow,
    Shape::kColumn / Policy::WarpShape::kColumn
  >;

  static_assert(!(ThreadShape::kColumn % Policy::LaneMmaShape::kN), 
    "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  /// Number of individual loads
  using Iterations = MatrixShape<
    ThreadShape::kRow,
    ThreadShape::kColumn / Policy::LaneMmaShape::kN
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;



  /// Operand tag
  static Operand const kOperand = Operand::kA;

  static_assert(!(Shape::kRow % Policy::WarpShape::kRow), 
    "The warp-level GEMM M size must be divisible by the number of threads arranged along the M dimension.");

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
  static_assert(Policy::WarpShape::kRow > 0, "Policy::WarpShape::kRow must be greater than zero.");
  static_assert(Shape::kRow / Policy::WarpShape::kRow > 0, "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");

  /// Thread-level shape of a fragment
  using ThreadShape = MatrixShape<
    Shape::kRow / Policy::WarpShape::kRow,
    Shape::kColumn
  >;

  static_assert(!(ThreadShape::kRow % Policy::LaneMmaShape::kM), 
    "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  /// Number of individual loads (scalar loads)
  using Iterations = MatrixShape<
    ThreadShape::kRow / Policy::LaneMmaShape::kM,
    ThreadShape::kColumn
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;



