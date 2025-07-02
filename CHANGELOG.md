# Changelog

## [0.2.1] - 2025-01-07

### Added
- **Parallelization**: Added parallel execution with progress bars to multiple commands:
  - `list-local`: Parallel port scanning and model detection
  - `list-remote`: Parallel SSH connections to multiple servers
  - `gpu-idle-top`: Parallel GPU stats collection for initial scan and live updates
  - `vllm-queue-top`: Parallel metrics collection for multiple vLLM instances
- **GPU count column**: Added GPU count information in `gpu-idle-top` command
- **Parallel utilities**: New `parallel_map_with_progress` utility for consistent parallel execution across commands

### Changed
- **SSH non-interactive mode**: All SSH commands now use `-o BatchMode=yes` to prevent password prompts that could hang parallel operations
- **GPU Idle Top improvements**: 
  - Removed "Mem Graph" column for cleaner output
  - Added "GPU count" column showing number of GPUs per host
  - Improved performance with parallel execution
- **vLLM Queue Top**: Increased default refresh interval from 1.0s to 10.0s for better performance

### Fixed
- **Terminal corruption**: Fixed issues where progress bars could leave terminal in corrupted state
- **SSH hanging**: Eliminated SSH password prompts that caused commands to hang during parallel execution
- **Progress bar cleanup**: Improved progress bar handling in live-updating commands

### Performance
- **Significantly faster execution** for commands dealing with multiple hosts/ports due to parallelization
- **Better responsiveness** in live-updating commands (`gpu-idle-top`, `vllm-queue-top`)

## [0.2.0] - 2025-06-19

### Added
- New `serve` command for remote deployment with flexible VLLM arguments.
- Shell autocompletion instructions in README.md.
- Support for additional options: tensor parallel size and remote port in `serve` command. 