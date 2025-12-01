//! # Configuration System for LongQLoRA
//!
//! Pydantic-inspired configuration with validation, defaults, and environment variable support
//! for the ultra-high performance LLM context extension system.

use crate::error::{Result, LongQLoRAError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Loss function types for training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossFunction {
    /// Standard cross-entropy loss
    CrossEntropy,
    /// Focal loss for imbalanced datasets
    Focal,
    /// KL divergence loss for distillation
    KLDiv,
}

/// Quantization types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// No quantization
    None,
    /// 8-bit quantization
    Int8,
    /// 4-bit quantization
    Int4,
    /// 4-bit Normal Float
    NF4,
}

/// Attention mechanism types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionType {
    /// Standard attention
    Standard,
    /// Flash attention for speed
    Flash,
    /// LongQLoRA grouped attention
    LongQLoRA,
}

/// Optimizer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Adam optimizer
    Adam,
    /// AdamW optimizer
    AdamW,
    /// Lion optimizer (more memory efficient)
    Lion,
    /// 8-bit Adam (memory efficient)
    Adam8Bit,
}

/// Learning rate scheduler types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LRSchedulerType {
    /// Constant learning rate
    Constant,
    /// Linear decay
    Linear,
    /// Cosine annealing
    Cosine,
    /// Warmup with constant
    WarmupConstant,
    /// Warmup with linear decay
    WarmupLinear,
}

/// Dataset types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetType {
    /// JSON lines format
    JsonLines,
    /// Parquet format (efficient columnar)
    Parquet,
    /// Arrow format
    Arrow,
    /// Streaming dataset
    Streaming,
}

/// Main training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Model configuration
    pub model: ModelConfig,
    /// Training hyperparameters
    pub training: TrainingHyperparams,
    /// LoRA configuration
    pub lora: LoRAConfig,
    /// Quantization settings
    pub quantization: QuantizationConfig,
    /// Data configuration
    pub data: DataConfig,
    /// Loss function settings
    pub loss: LossConfig,
    /// Optimizer configuration
    pub optimizer: OptimizerConfig,
    /// Learning rate scheduler
    pub scheduler: LRSchedulerConfig,
    /// Output and logging
    pub output: OutputConfig,
    /// Performance and hardware settings
    pub performance: PerformanceConfig,
    /// Distributed training settings
    pub distributed: DistributedConfig,
    /// Metrics and monitoring
    pub metrics: MetricsConfig,
}

/// Model architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name or path (HuggingFace or local)
    pub name_or_path: String,
    /// Model type (llama, gpt2, etc.)
    pub model_type: String,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Model hidden size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key/value heads (for GQA)
    pub num_key_value_heads: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Attention type to use
    pub attention_type: AttentionType,
    /// Use gradient checkpointing
    pub gradient_checkpointing: bool,
    /// Tie word embeddings
    pub tie_word_embeddings: bool,
}

/// Training hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHyperparams {
    /// Maximum training steps
    pub max_steps: usize,
    /// Per-device train batch size
    pub per_device_train_batch_size: usize,
    /// Per-device eval batch size
    pub per_device_eval_batch_size: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,
    /// Warmup steps
    pub warmup_steps: usize,
    /// Logging steps
    pub logging_steps: usize,
    /// Save steps
    pub save_steps: usize,
    /// Evaluation steps
    pub eval_steps: usize,
    /// Save total limit (keep only N checkpoints)
    pub save_total_limit: usize,
    /// Use mixed precision (FP16)
    pub fp16: bool,
    /// Use BF16 mixed precision
    pub bf16: bool,
    /// Data type for model weights
    pub torch_dtype: String,
    /// Random seed
    pub seed: u64,
}

/// LoRA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// LoRA rank
    pub rank: usize,
    /// LoRA alpha (scaling factor)
    pub alpha: f64,
    /// LoRA dropout rate
    pub dropout: f64,
    /// Target modules for LoRA adaptation
    pub target_modules: Vec<String>,
    /// Train bias parameters
    pub bias: String, // "none", "all", "lora_only"
    /// LoRA task type
    pub task_type: String,
    /// Modules to save (instead of LoRA)
    pub modules_to_save: Vec<String>,
    /// Train embedding layer
    pub train_embedding: bool,
    /// Train normalization layers
    pub train_norm: bool,
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Load model in 4-bit precision
    pub load_in_4bit: bool,
    /// Load model in 8-bit precision
    pub load_in_8bit: bool,
    /// 4-bit quantization type
    pub bnb_4bit_quant_type: QuantizationType,
    /// 4-bit compute dtype
    pub bnb_4bit_compute_dtype: String,
    /// 4-bit double quantization
    pub bnb_4bit_use_double_quant: bool,
    /// 8-bit threshold for quantization
    pub llm_int8_threshold: f64,
    /// 8-bit FP16 weight handling
    pub llm_int8_has_fp16_weight: bool,
}

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Training data files
    pub train_files: Vec<PathBuf>,
    /// Validation data files
    pub validation_files: Vec<PathBuf>,
    /// Dataset type
    pub dataset_type: DatasetType,
    /// Maximum sequence length for preprocessing
    pub max_seq_length: usize,
    /// Use streaming dataset (for large datasets)
    pub streaming: bool,
    /// Preprocessing batch size
    pub preprocessing_batch_size: usize,
    /// Number of preprocessing workers
    pub preprocessing_num_workers: usize,
    /// Shuffle buffer size (for streaming)
    pub shuffle_buffer_size: usize,
}

/// Loss function configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    /// Loss function type
    pub loss_function: LossFunction,
    /// Focal loss alpha parameter
    pub focal_alpha: f64,
    /// Focal loss gamma parameter
    pub focal_gamma: f64,
    /// KL divergence temperature
    pub temperature: f64,
    /// Label smoothing factor
    pub label_smoothing: f64,
    /// Ignore index for loss computation
    pub ignore_index: i64,
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Optimizer type
    pub optimizer_type: OptimizerType,
    /// Adam beta1
    pub adam_beta1: f64,
    /// Adam beta2
    pub adam_beta2: f64,
    /// Adam epsilon
    pub adam_epsilon: f64,
    /// Lion beta1
    pub lion_beta1: f64,
    /// Lion beta2
    pub lion_beta2: f64,
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRSchedulerConfig {
    /// Scheduler type
    pub scheduler_type: LRSchedulerType,
    /// Learning rate ratio for warmup
    pub warmup_ratio: f64,
    /// Number of warmup steps
    pub num_warmup_steps: usize,
    /// Number of cycles for cosine scheduler
    pub num_cycles: f64,
    /// Power for polynomial decay
    pub power: f64,
    /// Last epoch for scheduler
    pub last_epoch: i64,
}

/// Output and logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory for checkpoints and logs
    pub output_dir: PathBuf,
    /// Logging directory
    pub logging_dir: PathBuf,
    /// Log level
    pub log_level: String,
    /// Use TensorBoard logging
    pub report_to: Vec<String>,
    /// Run name for experiment tracking
    pub run_name: Option<String>,
    /// Save strategy
    pub save_strategy: String,
    /// Evaluation strategy
    pub evaluation_strategy: String,
}

/// Performance and hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of dataloader workers
    pub dataloader_num_workers: usize,
    /// Pin memory for DataLoader
    pub dataloader_pin_memory: bool,
    /// Non-blocking data transfer
    pub dataloader_non_blocking: bool,
    /// Prefetch factor for DataLoader
    pub dataloader_prefetch_factor: usize,
    /// Use CUDA graphs for optimization
    pub cuda_graphs: bool,
    /// Memory optimization level
    pub memory_optimization: String,
    /// Use XLA compilation (TPU)
    pub use_xla: bool,
    /// Use JIT compilation
    pub jit_compile: bool,
    /// Profile memory usage
    pub profile_memory: bool,
}

/// Distributed training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Use distributed training
    pub enabled: bool,
    /// Backend for distributed training
    pub backend: String,
    /// Number of processes per node
    pub nproc_per_node: usize,
    /// Master address for distributed training
    pub master_addr: String,
    /// Master port
    pub master_port: String,
    /// Node rank
    pub nnodes: usize,
    /// Node rank
    pub node_rank: usize,
    /// Local rank
    pub local_rank: usize,
    /// World size
    pub world_size: usize,
    /// DeepSpeed configuration file
    pub deepspeed_config: Option<PathBuf>,
    /// Use DeepSpeed
    pub use_deepspeed: bool,
}

/// Metrics and monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Collect training metrics
    pub collect_metrics: bool,
    /// Metrics logging interval
    pub log_metrics_interval: usize,
    /// Profile training performance
    pub profile_training: bool,
    /// Memory monitoring
    pub monitor_memory: bool,
    /// GPU utilization monitoring
    pub monitor_gpu: bool,
    /// Custom metrics to collect
    pub custom_metrics: Vec<String>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingHyperparams::default(),
            lora: LoRAConfig::default(),
            quantization: QuantizationConfig::default(),
            data: DataConfig::default(),
            loss: LossConfig::default(),
            optimizer: OptimizerConfig::default(),
            scheduler: LRSchedulerConfig::default(),
            output: OutputConfig::default(),
            performance: PerformanceConfig::default(),
            distributed: DistributedConfig::default(),
            metrics: MetricsConfig::default(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name_or_path: "NousResearch/Llama-2-7b-hf".to_string(),
            model_type: "llama".to_string(),
            max_seq_length: 4096,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            num_hidden_layers: 32,
            vocab_size: 32000,
            attention_type: AttentionType::LongQLoRA,
            gradient_checkpointing: true,
            tie_word_embeddings: false,
        }
    }
}

impl Default for TrainingHyperparams {
    fn default() -> Self {
        Self {
            max_steps: 1000,
            per_device_train_batch_size: 1,
            per_device_eval_batch_size: 1,
            gradient_accumulation_steps: 1,
            learning_rate: 2e-4,
            weight_decay: 0.0,
            max_grad_norm: 1.0,
            warmup_steps: 20,
            logging_steps: 50,
            save_steps: 500,
            eval_steps: 500,
            save_total_limit: 3,
            fp16: true,
            bf16: false,
            torch_dtype: "float16".to_string(),
            seed: 42,
        }
    }
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 64,
            alpha: 16.0,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            bias: "none".to_string(),
            task_type: "CAUSAL_LM".to_string(),
            modules_to_save: vec![],
            train_embedding: false,
            train_norm: false,
        }
    }
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            load_in_4bit: true,
            load_in_8bit: false,
            bnb_4bit_quant_type: QuantizationType::NF4,
            bnb_4bit_compute_dtype: "float16".to_string(),
            bnb_4bit_use_double_quant: true,
            llm_int8_threshold: 6.0,
            llm_int8_has_fp16_weight: false,
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            train_files: vec![],
            validation_files: vec![],
            dataset_type: DatasetType::JsonLines,
            max_seq_length: 4096,
            streaming: false,
            preprocessing_batch_size: 1000,
            preprocessing_num_workers: 4,
            shuffle_buffer_size: 10000,
        }
    }
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            loss_function: LossFunction::CrossEntropy,
            focal_alpha: 1.0,
            focal_gamma: 2.0,
            temperature: 2.0,
            label_smoothing: 0.0,
            ignore_index: -100,
        }
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::AdamW,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            lion_beta1: 0.95,
            lion_beta2: 0.98,
        }
    }
}

impl Default for LRSchedulerConfig {
    fn default() -> Self {
        Self {
            scheduler_type: LRSchedulerType::WarmupConstant,
            warmup_ratio: 0.1,
            num_warmup_steps: 20,
            num_cycles: 0.5,
            power: 1.0,
            last_epoch: -1,
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./output"),
            logging_dir: PathBuf::from("./logs"),
            log_level: "info".to_string(),
            report_to: vec!["tensorboard".to_string()],
            run_name: None,
            save_strategy: "steps".to_string(),
            evaluation_strategy: "steps".to_string(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            dataloader_num_workers: 0,
            dataloader_pin_memory: true,
            dataloader_non_blocking: true,
            dataloader_prefetch_factor: 2,
            cuda_graphs: false,
            memory_optimization: "auto".to_string(),
            use_xla: false,
            jit_compile: false,
            profile_memory: false,
        }
    }
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: "nccl".to_string(),
            nproc_per_node: 1,
            master_addr: "127.0.0.1".to_string(),
            master_port: "29500".to_string(),
            nnodes: 1,
            node_rank: 0,
            local_rank: 0,
            world_size: 1,
            deepspeed_config: None,
            use_deepspeed: false,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            collect_metrics: true,
            log_metrics_interval: 10,
            profile_training: false,
            monitor_memory: true,
            monitor_gpu: true,
            custom_metrics: vec![],
        }
    }
}

impl TrainingConfig {
    /// Load configuration from YAML file
    pub fn from_yaml<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: TrainingConfig = serde_yaml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from JSON file
    pub fn from_json<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: TrainingConfig = serde_json::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to YAML file
    pub fn to_yaml<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let yaml = serde_yaml::to_string(self)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }

    /// Save configuration to JSON file
    pub fn to_json<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Model validation
        if self.model.max_seq_length == 0 {
            return Err(LongQLoRAError::ConfigError("max_seq_length must be > 0".to_string()));
        }

        // Training validation
        if self.training.max_steps == 0 {
            return Err(LongQLoRAError::ConfigError("max_steps must be > 0".to_string()));
        }

        if self.training.learning_rate <= 0.0 {
            return Err(LongQLoRAError::ConfigError("learning_rate must be > 0".to_string()));
        }

        // LoRA validation
        if self.lora.rank == 0 {
            return Err(LongQLoRAError::ConfigError("LoRA rank must be > 0".to_string()));
        }

        // Data validation
        if self.data.train_files.is_empty() {
            return Err(LongQLoRAError::ConfigError("train_files cannot be empty".to_string()));
        }

        Ok(())
    }

    /// Merge with environment variables
    pub fn merge_env(&mut self) -> Result<()> {
        // Merge environment variables for common settings
        if let Ok(output_dir) = std::env::var("LONGQLORA_OUTPUT_DIR") {
            self.output.output_dir = PathBuf::from(output_dir);
        }

        if let Ok(log_level) = std::env::var("LONGQLORA_LOG_LEVEL") {
            self.output.log_level = log_level;
        }

        if let Ok(seed) = std::env::var("LONGQLORA_SEED") {
            if let Ok(seed_val) = seed.parse::<u64>() {
                self.training.seed = seed_val;
            }
        }

        // Distributed training environment variables
        if let Ok(local_rank) = std::env::var("LOCAL_RANK") {
            if let Ok(rank) = local_rank.parse::<usize>() {
                self.distributed.local_rank = rank;
                self.distributed.enabled = true;
            }
        }

        if let Ok(world_size) = std::env::var("WORLD_SIZE") {
            if let Ok(size) = world_size.parse::<usize>() {
                self.distributed.world_size = size;
            }
        }

        Ok(())
    }
}

/// Configuration builder for fluent API
pub struct ConfigBuilder {
    config: TrainingConfig,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: TrainingConfig::default(),
        }
    }

    pub fn model<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut ModelConfig),
    {
        f(&mut self.config.model);
        self
    }

    pub fn training<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut TrainingHyperparams),
    {
        f(&mut self.config.training);
        self
    }

    pub fn lora<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut LoRAConfig),
    {
        f(&mut self.config.lora);
        self
    }

    pub fn build(self) -> Result<TrainingConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration validation utilities
pub mod validation {
    use super::*;

    pub fn validate_model_config(config: &ModelConfig) -> Result<()> {
        if config.hidden_size == 0 {
            return Err(LongQLoRAError::ConfigError("hidden_size must be > 0".to_string()));
        }

        if config.num_attention_heads == 0 {
            return Err(LongQLoRAError::ConfigError("num_attention_heads must be > 0".to_string()));
        }

        if config.num_key_value_heads > config.num_attention_heads {
            return Err(LongQLoRAError::ConfigError(
                "num_key_value_heads cannot exceed num_attention_heads".to_string()
            ));
        }

        Ok(())
    }

    pub fn validate_training_config(config: &TrainingHyperparams) -> Result<()> {
        if config.per_device_train_batch_size == 0 {
            return Err(LongQLoRAError::ConfigError("per_device_train_batch_size must be > 0".to_string()));
        }

        if config.learning_rate <= 0.0 || config.learning_rate > 1.0 {
            return Err(LongQLoRAError::ConfigError("learning_rate must be in (0, 1]".to_string()));
        }

        if config.weight_decay < 0.0 {
            return Err(LongQLoRAError::ConfigError("weight_decay must be >= 0".to_string()));
        }

        Ok(())
    }
}
