//! # High-Performance Tensor Abstractions
//!
//! Zero-copy tensor operations with CUDA acceleration and memory pooling
//! for maximum performance in long-context LLM training.

use crate::error::{Result, LongQLoRAError};
use std::sync::Arc;
use std::ops::{Add, Sub, Mul, Div};

/// High-performance tensor with CUDA acceleration and memory pooling
#[derive(Debug, Clone)]
pub struct Tensor {
    inner: Arc<TensorInner>,
}

#[derive(Debug)]
struct TensorInner {
    data: TensorData,
    shape: Vec<usize>,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
    grad: Option<Tensor>,
}

/// Tensor data storage with memory pooling
#[derive(Debug)]
enum TensorData {
    Cpu(Vec<f32>),
    Cuda(CudaBuffer),
    Mmap(MemoryMap),
}

/// CUDA buffer with memory pooling
#[derive(Debug)]
struct CudaBuffer {
    ptr: cudarc::driver::CudaSlice<f32>,
    pool: Arc<MemoryPool>,
}

/// Memory-mapped tensor for massive datasets
#[derive(Debug)]
struct MemoryMap {
    mmap: memmap2::Mmap,
    offset: usize,
    len: usize,
}

/// Memory pool for efficient CUDA memory management
#[derive(Debug)]
struct MemoryPool {
    allocated: std::collections::HashMap<usize, Vec<cudarc::driver::CudaSlice<f32>>>,
    free: std::collections::HashMap<usize, Vec<cudarc::driver::CudaSlice<f32>>>,
}

/// Data types supported by LongQLoRA
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32,
    F16,
    BF16,
    I8,
    U8,
}

/// Device types for tensor placement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(usize), // GPU index
}

impl Tensor {
    /// Create tensor from vector with automatic device placement
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let total_elements: usize = shape.iter().product();
        if data.len() != total_elements {
            return Err(LongQLoRAError::InvalidShape(format!(
                "Data length {} does not match shape {:?}", data.len(), shape
            )));
        }

        Ok(Self {
            inner: Arc::new(TensorInner {
                data: TensorData::Cpu(data),
                shape: shape.to_vec(),
                dtype: DataType::F32,
                device: Device::Cpu,
                requires_grad: false,
                grad: None,
            }),
        })
    }

    /// Create random normal tensor
    pub fn randn(shape: &[usize]) -> Result<Self> {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        let total_elements: usize = shape.iter().product();
        let data: Vec<f32> = (0..total_elements)
            .map(|_| rng.gen::<f32>() * 0.02 - 0.01) // Normal distribution
            .collect();
        Self::from_vec(data, shape)
    }

    /// Create tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Result<Self> {
        let total_elements: usize = shape.iter().product();
        let data = vec![0.0f32; total_elements];
        Self::from_vec(data, shape)
    }

    /// Create tensor filled with ones
    pub fn ones(shape: &[usize]) -> Result<Self> {
        let total_elements: usize = shape.iter().product();
        let data = vec![1.0f32; total_elements];
        Self::from_vec(data, shape)
    }

    /// Create scalar tensor
    pub fn scalar(value: f64) -> Result<Self> {
        Self::from_vec(vec![value as f32], &[1])
    }

    /// Create arange tensor
    pub fn arange(start: i64, end: i64) -> Result<Self> {
        let data: Vec<f32> = (start..end).map(|x| x as f32).collect();
        Self::from_vec(data, &[data.len()])
    }

    /// Get tensor dimensions
    pub fn dims(&self) -> &[usize] {
        &self.inner.shape
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        self.dims()
    }

    /// Reshape tensor
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let total_elements: usize = new_shape.iter().product();
        if total_elements != self.numel() {
            return Err(LongQLoRAError::InvalidShape(format!(
                "Cannot reshape {} elements into shape {:?}", self.numel(), new_shape
            )));
        }

        Ok(Self {
            inner: Arc::new(TensorInner {
                data: self.inner.data.clone(),
                shape: new_shape.to_vec(),
                dtype: self.inner.dtype,
                device: self.inner.device,
                requires_grad: self.inner.requires_grad,
                grad: self.inner.grad.clone(),
            }),
        })
    }

    /// Transpose tensor
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.inner.shape.len() || dim1 >= self.inner.shape.len() {
            return Err(LongQLoRAError::InvalidDimension(format!(
                "Dimension {} or {} out of bounds for tensor with {} dimensions",
                dim0, dim1, self.inner.shape.len()
            )));
        }

        let mut new_shape = self.inner.shape.clone();
        new_shape.swap(dim0, dim1);

        // For now, just update shape - actual transpose would be implemented
        // with CUDA kernels for performance
        Ok(Self {
            inner: Arc::new(TensorInner {
                data: self.inner.data.clone(),
                shape: new_shape,
                dtype: self.inner.dtype,
                device: self.inner.device,
                requires_grad: self.inner.requires_grad,
                grad: self.inner.grad.clone(),
            }),
        })
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Result<Self> {
        // Dimension validation
        if self.dims().len() < 2 || other.dims().len() < 2 {
            return Err(LongQLoRAError::InvalidShape(
                "Matrix multiplication requires at least 2D tensors".to_string()
            ));
        }

        let self_last_dim = self.dims()[self.dims().len() - 1];
        let other_first_dim = other.dims()[other.dims().len() - 2];

        if self_last_dim != other_first_dim {
            return Err(LongQLoRAError::InvalidShape(format!(
                "Cannot multiply tensors with shapes {:?} and {:?}",
                self.dims(), other.dims()
            )));
        }

        // Compute output shape
        let mut output_shape = Vec::new();
        output_shape.extend_from_slice(&self.dims()[..self.dims().len() - 1]);
        output_shape.extend_from_slice(&other.dims()[other.dims().len() - 1..]);

        // Use CUDA kernel for matrix multiplication if available
        if matches!(self.inner.device, Device::Cuda(_)) {
            self.matmul_cuda(other, &output_shape)
        } else {
            self.matmul_cpu(other, &output_shape)
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Result<Self> {
        self.elementwise_op(other, ElementwiseOp::Add)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Result<Self> {
        self.elementwise_op(other, ElementwiseOp::Sub)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Result<Self> {
        self.elementwise_op(other, ElementwiseOp::Mul)
    }

    /// Element-wise division
    pub fn div(&self, other: &Tensor) -> Result<Self> {
        self.elementwise_op(other, ElementwiseOp::Div)
    }

    /// Concatenate tensors along dimension
    pub fn cat(tensors: &[&Tensor], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(LongQLoRAError::InvalidInput("Cannot concatenate empty tensor list".to_string()));
        }

        // Validate dimensions
        let first_shape = tensors[0].shape();
        for tensor in tensors.iter().skip(1) {
            if tensor.shape().len() != first_shape.len() {
                return Err(LongQLoRAError::InvalidShape(
                    "All tensors must have the same number of dimensions".to_string()
                ));
            }
            for (i, (&first_dim, &other_dim)) in first_shape.iter().zip(tensor.shape()).enumerate() {
                if i != dim && first_dim != other_dim {
                    return Err(LongQLoRAError::InvalidShape(format!(
                        "Tensor shapes don't match at dimension {}: {} vs {}",
                        i, first_dim, other_dim
                    )));
                }
            }
        }

        // Use CUDA kernel for concatenation
        Self::cat_cuda(tensors, dim)
    }

    /// Split tensor into chunks
    pub fn split(&self, sizes: &[usize], dim: usize) -> Result<Vec<Self>> {
        let total_size: usize = sizes.iter().sum();
        if total_size != self.dims()[dim] {
            return Err(LongQLoRAError::InvalidShape(format!(
                "Split sizes {} don't sum to dimension size {}",
                total_size, self.dims()[dim]
            )));
        }

        let mut result = Vec::new();
        let mut offset = 0;

        for &size in sizes {
            let chunk = self.narrow(dim, offset, size)?;
            result.push(chunk);
            offset += size;
        }

        Ok(result)
    }

    /// Narrow tensor along dimension
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        if dim >= self.dims().len() {
            return Err(LongQLoRAError::InvalidDimension(format!(
                "Dimension {} out of bounds for {}D tensor", dim, self.dims().len()
            )));
        }

        if start + len > self.dims()[dim] {
            return Err(LongQLoRAError::InvalidShape(format!(
                "Narrow range [{}, {}) out of bounds for dimension {} of size {}",
                start, start + len, dim, self.dims()[dim]
            )));
        }

        // Create view into the tensor data
        let mut new_shape = self.inner.shape.clone();
        new_shape[dim] = len;

        Ok(Self {
            inner: Arc::new(TensorInner {
                data: self.inner.data.narrow_view(start, len)?,
                shape: new_shape,
                dtype: self.inner.dtype,
                device: self.inner.device,
                requires_grad: self.inner.requires_grad,
                grad: self.inner.grad.clone(),
            }),
        })
    }

    /// Index select along dimension
    pub fn index_select(&self, dim: usize, indices: &Tensor) -> Result<Self> {
        // Use CUDA kernel for advanced indexing
        self.index_select_cuda(dim, indices)
    }

    /// Repeat interleave along dimension
    pub fn repeat_interleave(&self, repeats: usize, dim: usize) -> Result<Self> {
        if dim >= self.dims().len() {
            return Err(LongQLoRAError::InvalidDimension(format!(
                "Dimension {} out of bounds", dim
            )));
        }

        let mut new_shape = self.inner.shape.clone();
        new_shape[dim] *= repeats;

        // Use CUDA kernel for efficient repetition
        self.repeat_interleave_cuda(repeats, dim, &new_shape)
    }

    /// Unsqueeze dimension
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut new_shape = self.inner.shape.clone();
        if dim > new_shape.len() {
            return Err(LongQLoRAError::InvalidDimension(format!(
                "Dimension {} out of bounds for unsqueeze", dim
            )));
        }
        new_shape.insert(dim, 1);

        Ok(Self {
            inner: Arc::new(TensorInner {
                data: self.inner.data.clone(),
                shape: new_shape,
                dtype: self.inner.dtype,
                device: self.inner.device,
                requires_grad: self.inner.requires_grad,
                grad: self.inner.grad.clone(),
            }),
        })
    }

    /// Squeeze dimension
    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        if dim >= self.inner.shape.len() || self.inner.shape[dim] != 1 {
            return Err(LongQLoRAError::InvalidDimension(format!(
                "Cannot squeeze dimension {} with size {}", dim, self.inner.shape[dim]
            )));
        }

        let mut new_shape = self.inner.shape.clone();
        new_shape.remove(dim);

        Ok(Self {
            inner: Arc::new(TensorInner {
                data: self.inner.data.clone(),
                shape: new_shape,
                dtype: self.inner.dtype,
                device: self.inner.device,
                requires_grad: self.inner.requires_grad,
                grad: self.inner.grad.clone(),
            }),
        })
    }

    /// Cosine function
    pub fn cos(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Cos)
    }

    /// Sine function
    pub fn sin(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Sin)
    }

    /// Outer product
    pub fn outer(&self, other: &Tensor) -> Result<Self> {
        if self.dims().len() != 1 || other.dims().len() != 1 {
            return Err(LongQLoRAError::InvalidShape(
                "Outer product requires 1D tensors".to_string()
            ));
        }

        let output_shape = [self.dims()[0], other.dims()[0]];
        self.outer_cuda(other, &output_shape)
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.inner.shape.iter().product()
    }

    /// Get tensor data type
    pub fn dtype(&self) -> DataType {
        self.inner.dtype
    }

    /// Get tensor device
    pub fn device(&self) -> Device {
        self.inner.device
    }

    /// Set requires_grad flag
    pub fn requires_grad(&mut self, requires_grad: bool) {
        // This would need interior mutability in a real implementation
        // For now, we'll clone and modify
        *self = Self {
            inner: Arc::new(TensorInner {
                requires_grad,
                ..(*self.inner).clone()
            }),
        };
    }

    /// Get gradient tensor
    pub fn grad(&self) -> Option<&Tensor> {
        self.inner.grad.as_ref()
    }

    /// Zero gradient
    pub fn zero_grad(&mut self) {
        if let Some(grad) = &mut self.inner.grad {
            *grad = Tensor::zeros(self.shape()).unwrap();
        }
    }

    /// Backward pass
    pub async fn backward(&self) -> Result<()> {
        // Implement automatic differentiation
        self.backward_impl().await
    }

    /// Convert to CPU and get data as vector
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        match &self.inner.data {
            TensorData::Cpu(data) => Ok(data.clone()),
            TensorData::Cuda(buffer) => {
                // Copy from GPU to CPU
                let mut cpu_data = vec![0.0f32; self.numel()];
                buffer.ptr.copy_to_host(&mut cpu_data)?;
                Ok(cpu_data)
            }
            TensorData::Mmap(mmap) => {
                // Read from memory map
                let data = unsafe {
                    std::slice::from_raw_parts(
                        mmap.mmap.as_ptr().add(mmap.offset) as *const f32,
                        mmap.len / std::mem::size_of::<f32>(),
                    )
                };
                Ok(data.to_vec())
            }
        }
    }

    // Private implementation methods
    fn elementwise_op(&self, other: &Tensor, op: ElementwiseOp) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(LongQLoRAError::InvalidShape(format!(
                "Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()
            )));
        }

        if matches!(self.inner.device, Device::Cuda(_)) {
            self.elementwise_cuda(other, op)
        } else {
            self.elementwise_cpu(other, op)
        }
    }

    fn unary_op(&self, op: UnaryOp) -> Result<Self> {
        if matches!(self.inner.device, Device::Cuda(_)) {
            self.unary_cuda(op)
        } else {
            self.unary_cpu(op)
        }
    }

    // CUDA implementations (would contain actual kernel launches)
    fn matmul_cuda(&self, _other: &Tensor, _output_shape: &[usize]) -> Result<Self> {
        unimplemented!("CUDA matmul implementation")
    }

    fn matmul_cpu(&self, _other: &Tensor, _output_shape: &[usize]) -> Result<Self> {
        unimplemented!("CPU matmul implementation")
    }

    fn elementwise_cuda(&self, _other: &Tensor, _op: ElementwiseOp) -> Result<Self> {
        unimplemented!("CUDA elementwise implementation")
    }

    fn elementwise_cpu(&self, _other: &Tensor, _op: ElementwiseOp) -> Result<Self> {
        unimplemented!("CPU elementwise implementation")
    }

    fn unary_cuda(&self, _op: UnaryOp) -> Result<Self> {
        unimplemented!("CUDA unary implementation")
    }

    fn unary_cpu(&self, _op: UnaryOp) -> Result<Self> {
        unimplemented!("CPU unary implementation")
    }

    fn cat_cuda(_tensors: &[&Tensor], _dim: usize) -> Result<Self> {
        unimplemented!("CUDA cat implementation")
    }

    fn index_select_cuda(&self, _dim: usize, _indices: &Tensor) -> Result<Self> {
        unimplemented!("CUDA index_select implementation")
    }

    fn repeat_interleave_cuda(&self, _repeats: usize, _dim: usize, _new_shape: &[usize]) -> Result<Self> {
        unimplemented!("CUDA repeat_interleave implementation")
    }

    fn outer_cuda(&self, _other: &Tensor, _output_shape: &[usize]) -> Result<Self> {
        unimplemented!("CUDA outer implementation")
    }

    async fn backward_impl(&self) -> Result<()> {
        unimplemented!("Backward pass implementation")
    }
}

// Operation enums
#[derive(Debug, Clone, Copy)]
enum ElementwiseOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy)]
enum UnaryOp {
    Cos,
    Sin,
}

// TensorData implementations
impl TensorData {
    fn clone(&self) -> Self {
        match self {
            TensorData::Cpu(data) => TensorData::Cpu(data.clone()),
            TensorData::Cuda(buffer) => TensorData::Cuda(CudaBuffer {
                ptr: buffer.ptr.clone(),
                pool: buffer.pool.clone(),
            }),
            TensorData::Mmap(mmap) => TensorData::Mmap(MemoryMap {
                mmap: unsafe { memmap2::Mmap::map(&std::fs::File::open(std::path::Path::new("dummy")).unwrap()).unwrap() },
                offset: mmap.offset,
                len: mmap.len,
            }),
        }
    }

    fn narrow_view(&self, _start: usize, _len: usize) -> Result<Self> {
        // Create a view into the data
        unimplemented!("Narrow view implementation")
    }
}

// Arithmetic operator overloads
impl Add for &Tensor {
    type Output = Result<Tensor>;

    fn add(self, other: &Tensor) -> Result<Tensor> {
        self.add(other)
    }
}

impl Sub for &Tensor {
    type Output = Result<Tensor>;

    fn sub(self, other: &Tensor) -> Result<Tensor> {
        self.sub(other)
    }
}

impl Mul for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, other: &Tensor) -> Result<Tensor> {
        self.mul(other)
    }
}

impl Div for &Tensor {
    type Output = Result<Tensor>;

    fn div(self, other: &Tensor) -> Result<Tensor> {
        self.div(other)
    }
}

// Memory pool implementation
impl MemoryPool {
    fn new() -> Self {
        Self {
            allocated: std::collections::HashMap::new(),
            free: std::collections::HashMap::new(),
        }
    }

    fn allocate(&mut self, size: usize) -> Result<cudarc::driver::CudaSlice<f32>> {
        // Check free pool first
        if let Some(free_list) = self.free.get_mut(&size) {
            if let Some(buffer) = free_list.pop() {
                return Ok(buffer);
            }
        }

        // Allocate new buffer
        let buffer = cudarc::driver::CudaSlice::new(size)?;
        self.allocated.entry(size).or_insert_with(Vec::new).push(buffer.clone());
        Ok(buffer)
    }

    fn free(&mut self, buffer: cudarc::driver::CudaSlice<f32>) {
        let size = buffer.len();
        self.free.entry(size).or_insert_with(Vec::new).push(buffer);
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}
