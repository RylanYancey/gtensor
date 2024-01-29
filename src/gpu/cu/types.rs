
//! 6.1 Data types used by CUDA driver

use super::sys;

#[derive(Copy, Clone)]
pub struct Array {
    pub(crate) ptr: sys::CUarray,
}

#[derive(Copy, Clone)]
pub struct Context {
    pub(crate) ptr: sys::CUcontext,
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

#[derive(Copy, Clone)]
pub struct Device {
    pub(crate) ptr: sys::CUdevice,
}

#[derive(Copy, Clone)]
pub struct DevicePtr {
    pub(crate) ptr: sys::CUdeviceptr,
}

#[derive(Copy, Clone)]
pub struct Event {
    pub(crate) ptr: sys::CUevent,
}

#[derive(Copy, Clone)]
pub struct ExternalMemory {
    pub(crate) ptr: sys::CUexternalMemory,
}

#[derive(Copy, Clone)]
pub struct Function {
    pub(crate) ptr: sys::CUfunction,
}

#[derive(Copy, Clone)]
pub struct Graph {
    pub(crate) ptr: sys::CUgraph,
}

#[derive(Copy, Clone)]
pub struct GraphExec {
    pub(crate) ptr: sys::CUgraphExec,
}

#[derive(Copy, Clone)]
pub struct GraphNode {
    pub(crate) ptr: sys::CUgraphNode,
}

#[derive(Copy, Clone)]
pub struct GraphicsResource {
    pub(crate) ptr: sys::CUgraphicsResource,
}

#[derive(Copy, Clone)]
pub struct KernelNodeAttrID {
    pub(crate) ptr: sys::CUkernelNodeAttrID
}

#[derive(Copy, Clone)]
pub struct MemoryPool {
    pub(crate) ptr: sys::CUmemoryPool,
}

#[derive(Copy, Clone)]
pub struct MipmappedArray {
    pub(crate) ptr: sys::CUmipmappedArray,
}

#[derive(Copy, Clone)]
pub struct Module {
    pub(crate) ptr: sys::CUmodule,
}

unsafe impl Send for Module { }
unsafe impl Sync for Module { }

#[derive(Copy, Clone)]
pub struct Stream {
    pub(crate) ptr: sys::CUstream,
}

impl Stream {
    pub fn null() -> Self {
        Self {
            ptr: std::ptr::null_mut()
        }
    }
}

#[derive(Copy, Clone)]
pub struct StreamAttrID {
    pub(crate) ptr: sys::CUstreamAttrID,
}

#[derive(Copy, Clone)]
pub struct StreamAttrValue {
    pub(crate) ptr: sys::CUstreamAttrValue_v1,
}

#[derive(Copy, Clone)]
pub struct SurfObject {
    pub(crate) ptr: sys::CUsurfObject,
}

#[derive(Copy, Clone)]
pub struct SurfRef {
    pub(crate) ptr: sys::CUsurfref,
}

#[derive(Copy, Clone)]
pub struct TexObject {
    pub(crate) ptr: sys::CUtexObject,
}

#[derive(Copy, Clone)]
pub struct TexRef {
    pub(crate) ptr: sys::CUtexref,
}

#[derive(Copy, Clone)]
pub struct UserObject {
    pub(crate) ptr: sys::CUuserObject,
}

/// Device Attributes queriable with device_get_attribute.
pub enum DeviceAttribute {
    /// Maximum number of threads per bloc
    MAX_THREADS_PER_BLOCK = 1,
    /// Maximum block dimension 
    MAX_BLOCK_DIM_X = 2,
    /// Maximum block dimension 
    MAX_BLOCK_DIM_Y = 3,
    /// Maximum block dimension 
    MAX_BLOCK_DIM_Z = 4,
    /// Maximum grid dimension 
    MAX_GRID_DIM_X = 5,
    /// Maximum grid dimension 
    MAX_GRID_DIM_Y = 6,
    /// Maximum grid dimension 
    MAX_GRID_DIM_Z = 7,
    /// Maximum shared memory available per block in byte
    MAX_SHARED_MEMORY_PER_BLOCK = 8,
    /// Memory available on device for __constant__ variables in a CUDA C kernel in byte
    TOTAL_CONSTANT_MEMORY = 9,
    /// Warp size in thread
    WARP_SIZE = 10,
    /// Maximum pitch in bytes allowed by memory copie
    MAX_PITCH = 11,
    /// Maximum number of 32-bit registers available per bloc
    MAX_REGISTERS_PER_BLOCK = 12,
    /// Typical clock frequency in kilohert
    CLOCK_RATE = 13,
    /// Alignment requirement for texture
    TEXTURE_ALIGNMENT = 14,
    /// Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
    GPU_OVERLAP = 15,
    /// Number of multiprocessors on devic
    MULTIPROCESSOR_COUNT = 16,
    /// Specifies whether there is a run time limit on kernel
    KERNEL_EXEC_TIMEOUT = 17,
    /// Device is integrated with host memor
    INTEGRATED = 18,
    /// Device can map host memory into CUDA address spac
    CAN_MAP_HOST_MEMORY = 19,
    /// Compute mode (See ::CUcomputemode for details
    COMPUTE_MODE = 20,
    /// Maximum 1D texture widt
    MAXIMUM_TEXTURE1D_WIDTH = 21,
    /// Maximum 2D texture widt
    MAXIMUM_TEXTURE2D_WIDTH = 22,
    /// Maximum 2D texture heigh
    MAXIMUM_TEXTURE2D_HEIGHT = 23,
    /// Maximum 3D texture widt
    MAXIMUM_TEXTURE3D_WIDTH = 24,
    /// Maximum 3D texture heigh
    MAXIMUM_TEXTURE3D_HEIGHT = 25,
    /// Maximum 3D texture dept
    MAXIMUM_TEXTURE3D_DEPTH = 26,
    /// Maximum 2D layered texture widt
    MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    /// Maximum 2D layered texture heigh
    MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    /// Maximum layers in a 2D layered textur
    MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    /// Alignment requirement for surface
    SURFACE_ALIGNMENT = 30,
    /// Device can possibly execute multiple kernels concurrentl
    CONCURRENT_KERNELS = 31,
    /// Device has ECC support enable
    ECC_ENABLED = 32,
    /// PCI bus ID of the devic
    PCI_BUS_ID = 33,
    /// PCI device ID of the devic
    PCI_DEVICE_ID = 34,
    /// Device is using TCC driver mode
    TCC_DRIVER = 35,
    /// Peak memory clock frequency in kilohert
    MEMORY_CLOCK_RATE = 36,
    /// Global memory bus width in bit
    GLOBAL_MEMORY_BUS_WIDTH = 37,
    /// Size of L2 cache in byte
    L2_CACHE_SIZE = 38,
    /// Maximum resident threads per multiprocesso
    MAX_THREADS_PER_MULTIPROCESSOR = 39,
    /// Number of asynchronous engine
    ASYNC_ENGINE_COUNT = 40,
    /// Device shares a unified address space with the hos
    UNIFIED_ADDRESSING = 41,
    /// Maximum 1D layered texture widt
    MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    /// Maximum layers in a 1D layered textur
    MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    /// Deprecated, do not use
    CAN_TEX2D_GATHER = 44,
    /// Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is se
    MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    /// Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is se
    MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    /// Alternate maximum 3D texture widt
    MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    /// Alternate maximum 3D texture heigh
    MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    /// Alternate maximum 3D texture dept
    MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    /// PCI domain ID of the devic
    PCI_DOMAIN_ID = 50,
    /// Pitch alignment requirement for texture
    TEXTURE_PITCH_ALIGNMENT = 51,
    /// Maximum cubemap texture width/heigh
    MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    /// Maximum cubemap layered texture width/heigh
    MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    /// Maximum layers in a cubemap layered textur
    MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    /// Maximum 1D surface widt
    MAXIMUM_SURFACE1D_WIDTH = 55,
    /// Maximum 2D surface widt
    MAXIMUM_SURFACE2D_WIDTH = 56,
    /// Maximum 2D surface heigh
    MAXIMUM_SURFACE2D_HEIGHT = 57,
    /// Maximum 3D surface widt
    MAXIMUM_SURFACE3D_WIDTH = 58,
    /// Maximum 3D surface heigh
    MAXIMUM_SURFACE3D_HEIGHT = 59,
    /// Maximum 3D surface dept
    MAXIMUM_SURFACE3D_DEPTH = 60,
    /// Maximum 1D layered surface widt
    MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    /// Maximum layers in a 1D layered surfac
    MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    /// Maximum 2D layered surface widt
    MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    /// Maximum 2D layered surface heigh
    MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    /// Maximum layers in a 2D layered surfac
    MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    /// Maximum cubemap surface widt
    MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    /// Maximum cubemap layered surface widt
    MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    /// Maximum layers in a cubemap layered surfac
    MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    /// Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead
    MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    /// Maximum 2D linear texture widt
    MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    /// Maximum 2D linear texture heigh
    MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    /// Maximum 2D linear texture pitch in byte
    MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    /// Maximum mipmapped 2D texture widt
    MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    /// Maximum mipmapped 2D texture heigh
    MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    /// Major compute capability version numbe
    COMPUTE_CAPABILITY_MAJOR = 75,
    /// Minor compute capability version numbe
    COMPUTE_CAPABILITY_MINOR = 76,
    /// Maximum mipmapped 1D texture widt
    MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    /// Device supports stream prioritie
    STREAM_PRIORITIES_SUPPORTED = 78,
    /// Device supports caching globals in L
    GLOBAL_L1_CACHE_SUPPORTED = 79,
    /// Device supports caching locals in L
    LOCAL_L1_CACHE_SUPPORTED = 80,
    /// Maximum shared memory available per multiprocessor in byte
    MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    /// Maximum number of 32-bit registers available per multiprocesso
    MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    /// Device can allocate managed memory on this syste
    MANAGED_MEMORY = 83,
    /// Device is on a multi-GPU boar
    MULTI_GPU_BOARD = 84,
    /// Unique id for a group of devices on the same multi-GPU boar
    MULTI_GPU_BOARD_GROUP_ID = 85,
    /// Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware
    HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    /// Ratio of single precision performance (in floating-point operations per second) to double precision performanc
    SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    /// Device supports coherently accessing pageable memory without calling cudaHostRegister on i
    PAGEABLE_MEMORY_ACCESS = 88,
    /// Device can coherently access managed memory concurrently with the CP
    CONCURRENT_MANAGED_ACCESS = 89,
    /// Device supports compute preemption
    COMPUTE_PREEMPTION_SUPPORTED = 90,
    /// Device can access host registered memory at the same virtual address as the CP
    CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    /// ::cuStreamBatchMemOp and related APIs are supported
    CAN_USE_STREAM_MEM_OPS = 92,
    /// 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs
    CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
    /// ::CU_STREAM_WAIT_VALUE_NOR is supported
    CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
    /// Device supports launching cooperative kernels via ::cuLaunchCooperativeKerne
    COOPERATIVE_LAUNCH = 95,
    /// Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated
    COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    /// Maximum optin shared memory per bloc
    MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    /// The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \\ref CUDA_MEMOP for additional details
    CAN_FLUSH_REMOTE_WRITES = 98,
    /// Device supports host memory registration via ::cudaHostRegister
    HOST_REGISTER_SUPPORTED = 99,
    /// Device accesses pageable memory via the host's page tables
    PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    /// The host can directly access managed memory on the device without migration
    DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    /// Device supports virtual memory management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related API
    VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    /// Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreat
    HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED ,
    /// Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreat
    HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    /// Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreat
    HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    /// Maximum number of blocks per multiprocesso
    MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    /// Device supports compression of memor
    GENERIC_COMPRESSION_SUPPORTED = 107,
    /// Maximum L2 persisting lines capacity setting in bytes
    MAX_PERSISTING_L2_CACHE_SIZE = 108,
    /// Maximum value of CUaccessPolicyWindow::num_bytes
    MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    /// Device supports specifying the GPUDirect RDMA flag with ::cuMemCreat
    GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    /// Shared memory reserved by CUDA driver per block in byte
    RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    /// Device supports sparse CUDA arrays and sparse CUDA mipmapped array
    SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    /// Device supports using the ::cuMemHostRegister flag ::CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GP
    READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    /// External timeline semaphore interop is supported on the devic
    TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    /// Device supports using the ::cuMemAllocAsync and ::cuMemPool family of API
    MEMORY_POOLS_SUPPORTED = 115,
    /// Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information
    GPU_DIRECT_RDMA_SUPPORTED = 116,
    /// The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions enu
    GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    /// GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here
    GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    /// Handle types supported with mempool based IP
    MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    /// Indicates device supports cluster launc
    CLUSTER_LAUNCH = 120,
    /// Device supports deferred mapping CUDA arrays and CUDA mipmapped array
    DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    /// 64-bit operations are supported in ::cuStreamBatchMemOp_v2 and related v2 MemOp APIs
    CAN_USE_64_BIT_STREAM_MEM_OPS_V2 = 122,
    /// ::CU_STREAM_WAIT_VALUE_NOR is supported by v2 MemOp APIs
    CAN_USE_STREAM_WAIT_VALUE_NOR_V2 = 123,
    /// Device supports buffer sharing with dma_buf mechanism
    DMA_BUF_SUPPORTED = 124,
    MAX = 125,
}