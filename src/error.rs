//! # Error Handling for LongQLoRA
//!
//! Comprehensive error types and handling for the ultra-high performance
//! LLM context extension system.

use std::fmt;

/// Result type alias for LongQLoRA operations
pub type Result<T> = std::result::Result<T, LongQLoRAError>;

/// Comprehensive error types for LongQLoRA operations
#[derive(Debug, thiserror::Error)]
pub enum LongQLoRAError {
    /// Invalid tensor shape or dimensions
    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),

    /// Invalid dimension for tensor operation
    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),

    /// Invalid input parameters
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// CUDA/GPU related errors
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Memory allocation failures
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),

    /// File I/O errors
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_yaml::Error),

    /// JSON serialization errors
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Model loading/parsing errors
    #[error("Model error: {0}")]
    ModelError(String),

    /// Training-related errors
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Configuration parsing errors
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Dataset processing errors
    #[error("Dataset error: {0}")]
    DatasetError(String),

    /// Tokenization errors
    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    /// Attention mechanism errors
    #[error("Attention error: {0}")]
    AttentionError(String),

    /// Gradient computation errors
    #[error("Gradient error: {0}")]
    GradientError(String),

    /// Optimization errors
    #[error("Optimization error: {0}")]
    OptimizationError(String),

    /// Distributed training errors
    #[error("Distributed training error: {0}")]
    DistributedError(String),

    /// Plugin/extension errors
    #[error("Plugin error: {0}")]
    PluginError(String),

    /// Generic errors with custom messages
    #[error("{0}")]
    Custom(String),
}

impl LongQLoRAError {
    /// Create a new custom error
    pub fn custom<S: Into<String>>(msg: S) -> Self {
        Self::Custom(msg.into())
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Recoverable errors (can retry or continue)
            Self::CudaError(_) | Self::MemoryError(_) | Self::IoError(_) => true,
            // Non-recoverable errors (fatal)
            Self::InvalidShape(_) | Self::InvalidDimension(_) | Self::InvalidInput(_) => false,
            // Context-dependent
            _ => false,
        }
    }

    /// Check if this is a CUDA-related error
    pub fn is_cuda_error(&self) -> bool {
        matches!(self, Self::CudaError(_))
    }

    /// Check if this is a memory-related error
    pub fn is_memory_error(&self) -> bool {
        matches!(self, Self::MemoryError(_))
    }

    /// Get error category for logging/metrics
    pub fn category(&self) -> &'static str {
        match self {
            Self::InvalidShape(_) | Self::InvalidDimension(_) => "validation",
            Self::InvalidInput(_) => "input",
            Self::CudaError(_) => "cuda",
            Self::MemoryError(_) => "memory",
            Self::IoError(_) => "io",
            Self::ModelError(_) => "model",
            Self::TrainingError(_) => "training",
            Self::ConfigError(_) => "config",
            Self::DatasetError(_) => "dataset",
            Self::TokenizationError(_) => "tokenization",
            Self::AttentionError(_) => "attention",
            Self::GradientError(_) => "gradient",
            Self::OptimizationError(_) => "optimization",
            Self::DistributedError(_) => "distributed",
            Self::PluginError(_) => "plugin",
            Self::SerdeError(_) | Self::JsonError(_) => "serialization",
            Self::Custom(_) => "custom",
        }
    }

    /// Convert to user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            Self::CudaError(msg) => format!("GPU error occurred: {}. Try restarting or using CPU mode.", msg),
            Self::MemoryError(msg) => format!("Out of memory: {}. Try reducing batch size or sequence length.", msg),
            Self::IoError(_) => "File I/O error occurred. Check file permissions and disk space.".to_string(),
            Self::InvalidShape(msg) => format!("Invalid tensor dimensions: {}", msg),
            Self::TrainingError(msg) => format!("Training failed: {}", msg),
            _ => self.to_string(),
        }
    }
}

/// Error context for better debugging
#[derive(Debug)]
pub struct ErrorContext {
    pub operation: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            timestamp: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Enhanced Result with context
pub type ContextResult<T> = std::result::Result<T, (LongQLoRAError, ErrorContext)>;

/// Extension trait for adding context to Results
pub trait ResultExt<T> {
    fn with_context<F>(self, f: F) -> ContextResult<T>
    where
        F: FnOnce() -> ErrorContext;

    fn context(self, operation: impl Into<String>) -> ContextResult<T>;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: Into<LongQLoRAError>,
{
    fn with_context<F>(self, f: F) -> ContextResult<T>
    where
        F: FnOnce() -> ErrorContext,
    {
        self.map_err(|e| (e.into(), f()))
    }

    fn context(self, operation: impl Into<String>) -> ContextResult<T> {
        self.map_err(|e| (e.into(), ErrorContext::new(operation)))
    }
}

/// Panic hook for better error reporting in production
pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|panic_info| {
        let backtrace = std::backtrace::Backtrace::capture();

        // Log panic information
        tracing::error!(
            "LongQLoRA panicked: {}",
            panic_info.payload().downcast_ref::<&str>().unwrap_or(&"Unknown panic")
        );

        if let Some(location) = panic_info.location() {
            tracing::error!("Panic location: {}:{}", location.file(), location.line());
        }

        tracing::error!("Backtrace:\n{}", backtrace);

        // In production, you might want to send this to a monitoring service
        // send_to_monitoring_service(panic_info, &backtrace);
    }));
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry the operation
    Retry { max_attempts: usize, backoff_ms: u64 },
    /// Fallback to CPU computation
    FallbackToCpu,
    /// Reduce batch size
    ReduceBatchSize { factor: f32 },
    /// Skip the current batch
    SkipBatch,
    /// Terminate training
    Terminate,
}

/// Error handler with recovery logic
#[derive(Debug)]
pub struct ErrorHandler {
    strategies: std::collections::HashMap<String, RecoveryStrategy>,
}

impl ErrorHandler {
    pub fn new() -> Self {
        let mut strategies = std::collections::HashMap::new();

        // Default recovery strategies
        strategies.insert(
            "cuda".to_string(),
            RecoveryStrategy::Retry { max_attempts: 3, backoff_ms: 1000 },
        );
        strategies.insert(
            "memory".to_string(),
            RecoveryStrategy::ReduceBatchSize { factor: 0.5 },
        );
        strategies.insert(
            "io".to_string(),
            RecoveryStrategy::Retry { max_attempts: 5, backoff_ms: 500 },
        );

        Self { strategies }
    }

    pub async fn handle_error(&self, error: &LongQLoRAError, context: &ErrorContext) -> RecoveryStrategy {
        let category = error.category();

        match self.strategies.get(category) {
            Some(strategy) => strategy.clone(),
            None => RecoveryStrategy::Terminate,
        }
    }
}

impl Default for ErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance monitoring for errors
#[derive(Debug)]
pub struct ErrorMetrics {
    error_counts: std::collections::HashMap<String, u64>,
    recovery_attempts: std::collections::HashMap<String, u64>,
    successful_recoveries: std::collections::HashMap<String, u64>,
}

impl ErrorMetrics {
    pub fn new() -> Self {
        Self {
            error_counts: std::collections::HashMap::new(),
            recovery_attempts: std::collections::HashMap::new(),
            successful_recoveries: std::collections::HashMap::new(),
        }
    }

    pub fn record_error(&mut self, error: &LongQLoRAError) {
        let category = error.category().to_string();
        *self.error_counts.entry(category).or_insert(0) += 1;
    }

    pub fn record_recovery_attempt(&mut self, category: &str) {
        *self.recovery_attempts.entry(category.to_string()).or_insert(0) += 1;
    }

    pub fn record_successful_recovery(&mut self, category: &str) {
        *self.successful_recoveries.entry(category.to_string()).or_insert(0) += 1;
    }

    pub fn get_stats(&self) -> std::collections::HashMap<String, serde_json::Value> {
        let mut stats = std::collections::HashMap::new();

        for (category, &count) in &self.error_counts {
            let attempts = self.recovery_attempts.get(category).copied().unwrap_or(0);
            let successes = self.successful_recoveries.get(category).copied().unwrap_or(0);
            let success_rate = if attempts > 0 { successes as f64 / attempts as f64 } else { 0.0 };

            stats.insert(category.clone(), serde_json::json!({
                "error_count": count,
                "recovery_attempts": attempts,
                "successful_recoveries": successes,
                "success_rate": success_rate
            }));
        }

        stats
    }
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for error handling
pub mod utils {
    use super::*;

    /// Convert any error to LongQLoRAError
    pub fn to_longqlora_error<E: std::error::Error + Send + Sync + 'static>(error: E) -> LongQLoRAError {
        LongQLoRAError::Custom(error.to_string())
    }

    /// Create a CUDA error
    pub fn cuda_error<S: Into<String>>(msg: S) -> LongQLoRAError {
        LongQLoRAError::CudaError(msg.into())
    }

    /// Create a memory error
    pub fn memory_error<S: Into<String>>(msg: S) -> LongQLoRAError {
        LongQLoRAError::MemoryError(msg.into())
    }

    /// Create a model error
    pub fn model_error<S: Into<String>>(msg: S) -> LongQLoRAError {
        LongQLoRAError::ModelError(msg.into())
    }

    /// Create a training error
    pub fn training_error<S: Into<String>>(msg: S) -> LongQLoRAError {
        LongQLoRAError::TrainingError(msg.into())
    }

    /// Wrap a result with additional context
    pub fn with_context<T, E, F>(
        result: std::result::Result<T, E>,
        context_fn: F,
    ) -> ContextResult<T>
    where
        E: Into<LongQLoRAError>,
        F: FnOnce() -> ErrorContext,
    {
        result.with_context(context_fn)
    }

    /// Log error with context
    pub fn log_error(error: &LongQLoRAError, context: &ErrorContext) {
        let level = if error.is_recoverable() {
            tracing::Level::WARN
        } else {
            tracing::Level::ERROR
        };

        tracing::event!(
            level,
            category = error.category(),
            operation = %context.operation,
            timestamp = %context.timestamp,
            metadata = ?context.metadata,
            "{:?}",
            error
        );
    }
}
