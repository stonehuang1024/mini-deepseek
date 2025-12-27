#!/bin/bash
# DeepSeek V3 Training and Inference Scripts
# ==========================================
#
# This script provides easy-to-use commands for:
# 1. Running tests
# 2. Pretraining
# 3. Supervised Fine-Tuning (SFT)
# 4. Reinforcement Learning (GRPO)
# 5. Inference
#
# Usage:
#   ./run.sh test           # Run all tests
#   ./run.sh pretrain       # Start pretraining
#   ./run.sh sft            # Start SFT
#   ./run.sh rl             # Start RL
#   ./run.sh inference      # Run inference demo
#   ./run.sh chat           # Interactive chat
#   ./run.sh tensorboard    # Start TensorBoard
#   ./run.sh full           # Run full pipeline (pretrain -> sft -> rl -> inference)
#


set -e

source ~/.zshrc 2>/dev/null || true

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
CONFIG="configs/config_default.yaml"
DEVICE="auto"

# Print header
print_header() {
    echo -e "${BLUE}"
    echo "======================================================================"
    echo "  DeepSeek V3 - Learning Implementation"
    echo "  Features: MLA, MoE, MTP, GRPO"
    echo "======================================================================"
    echo -e "${NC}"
}

# Print usage
print_usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  test          Run all tests"
    echo "  test-quick    Run quick model test"
    echo "  pretrain      Start pretraining (small dataset)"
    echo "  pretrain-large Start pretraining (large dataset ~10GB)"
    echo "  pretrain-test Quick pretrain test (50 steps)"
    echo "  sft           Start supervised fine-tuning"
    echo "  sft-test      Quick SFT test"
    echo "  rl            Start reinforcement learning (GRPO)"
    echo "  rl-test       Quick RL test"
    echo "  inference     Run inference demo"
    echo "  chat          Interactive chat mode"
    echo "  web-chat      Start web chat interface (ChatGPT-style)"
    echo "  tensorboard   Start TensorBoard server"
    echo "  full          Run full pipeline"
    echo "  full-test     Run full pipeline (quick test)"
    echo "  clean         Clean checkpoints and logs"
    echo "  help          Show this help message"
    echo ""
    echo "Options:"
    echo "  --config      Config file (default: config_default.yaml)"
    echo "  --device      Device: auto, cuda, mps, cpu (default: auto)"
    echo "  --checkpoint  Path to checkpoint for fine-tuning/inference"
    echo ""
    echo "Dataset Options (for pretrain):"
    echo "  --dataset_scale small   WikiText-2 (~13MB) - Fast for testing"
    echo "  --dataset_scale large   OpenWebText (~10GB) - For serious training"
}

# Check Python environment
check_environment() {
    echo -e "${YELLOW}Checking environment...${NC}"

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 is not installed${NC}"
        exit 1
    fi

    # Check required packages
    python3 -c "import torch" 2>/dev/null || {
        echo -e "${RED}Error: PyTorch not installed. Run: pip install torch${NC}"
        exit 1
    }

    python3 -c "import transformers" 2>/dev/null || {
        echo -e "${YELLOW}Warning: transformers not installed. Run: pip install transformers${NC}"
    }

    python3 -c "import datasets" 2>/dev/null || {
        echo -e "${YELLOW}Warning: datasets not installed. Run: pip install datasets${NC}"
    }

    echo -e "${GREEN}Environment OK${NC}"
}

# Run tests
run_tests() {
    echo -e "${BLUE}Running all tests...${NC}"
    python3 tests/test_all.py --test all
}

run_quick_test() {
    echo -e "${BLUE}Running quick model test...${NC}"
    python3 tests/test_all.py --test model
}

# Pretraining
run_pretrain() {
    echo -e "${BLUE}Starting pretraining (small dataset - WikiText-2)...${NC}"
    python3 train.py --mode pretrain --config "$CONFIG" --device "$DEVICE" --dataset_scale small "$@"
}

run_pretrain_large() {
    echo -e "${BLUE}Starting pretraining (large dataset - OpenWebText ~10GB)...${NC}"
    echo -e "${YELLOW}Warning: This will download ~10GB of data on first run!${NC}"
    python3 train.py --mode pretrain --config "$CONFIG" --device "$DEVICE" --dataset_scale large "$@"
}

run_pretrain_test() {
    echo -e "${BLUE}Running pretrain test (50 steps)...${NC}"
    python3 train.py --mode pretrain --config "$CONFIG" --device "$DEVICE" --dataset_scale small --test "$@"
}

# SFT
run_sft() {
    local checkpoint="${1:-checkpoints/pretrain/best.pt}"
    echo -e "${BLUE}Starting SFT from $checkpoint...${NC}"
    python3 train.py --mode sft --config "$CONFIG" --device "$DEVICE" --checkpoint "$checkpoint"
}

run_sft_test() {
    echo -e "${BLUE}Running SFT test...${NC}"
    python3 train.py --mode sft --config "$CONFIG" --device "$DEVICE" --test
}

# RL
run_rl() {
    local checkpoint="${1:-checkpoints/sft/best.pt}"
    echo -e "${BLUE}Starting RL (GRPO) from $checkpoint...${NC}"
    python3 train.py --mode rl --config "$CONFIG" --device "$DEVICE" --checkpoint "$checkpoint"
}

run_rl_test() {
    echo -e "${BLUE}Running RL test...${NC}"
    python3 train.py --mode rl --config "$CONFIG" --device "$DEVICE" --test
}

# Inference
run_inference() {
    local checkpoint="${1:-checkpoints/sft/best.pt}"
    echo -e "${BLUE}Running inference demo...${NC}"
    python3 deepseek/inference/inference.py --checkpoint "$checkpoint" --device "$DEVICE"
}

run_chat() {
    local checkpoint="${1:-checkpoints/sft/best.pt}"
    echo -e "${BLUE}Starting interactive chat...${NC}"
    python3 deepseek/inference/inference.py --checkpoint "$checkpoint" --device "$DEVICE" --interactive
}

run_web_chat() {
    echo -e "${BLUE}Starting web chat interface...${NC}"
    echo -e "${GREEN}Web chat will be available at: http://localhost:5001${NC}"
    python3 chat/app.py
}

# TensorBoard
run_tensorboard() {
    echo -e "${BLUE}Starting TensorBoard...${NC}"
    echo "View at: http://localhost:6006"
    tensorboard --logdir runs --port 6006
}

# Full pipeline
run_full_pipeline() {
    echo -e "${BLUE}Running full training pipeline...${NC}"
    echo ""

    echo -e "${YELLOW}Step 1/4: Pretraining${NC}"
    run_pretrain

    echo -e "${YELLOW}Step 2/4: SFT${NC}"
    run_sft "checkpoints/pretrain/best.pt"

    echo -e "${YELLOW}Step 3/4: RL (GRPO)${NC}"
    run_rl "checkpoints/sft/best.pt"

    echo -e "${YELLOW}Step 4/4: Inference${NC}"
    run_inference "checkpoints/rl/final.pt"

    echo -e "${GREEN}Full pipeline complete!${NC}"
}

run_full_pipeline_test() {
    echo -e "${BLUE}Running full pipeline test...${NC}"
    echo ""

    echo -e "${YELLOW}Step 1/4: Pretraining (test)${NC}"
    run_pretrain_test

    echo -e "${YELLOW}Step 2/4: SFT (test)${NC}"
    run_sft_test

    echo -e "${YELLOW}Step 3/4: RL (test)${NC}"
    run_rl_test

    echo -e "${YELLOW}Step 4/4: Inference${NC}"
    run_inference "checkpoints/pretrain/final.pt"

    echo -e "${GREEN}Full pipeline test complete!${NC}"
}

# Clean
clean() {
    echo -e "${YELLOW}Cleaning checkpoints and logs...${NC}"
    rm -rf checkpoints runs data/__pycache__ __pycache__
    echo -e "${GREEN}Cleaned!${NC}"
}

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Main command
COMMAND="${1:-help}"

print_header

case "$COMMAND" in
    test)
        check_environment
        run_tests
        ;;
    test-quick)
        check_environment
        run_quick_test
        ;;
    pretrain)
        check_environment
        shift
        run_pretrain "$@"
        ;;
    pretrain-large)
        check_environment
        shift
        run_pretrain_large "$@"
        ;;
    pretrain-test)
        check_environment
        shift
        run_pretrain_test "$@"
        ;;
    sft)
        check_environment
        shift
        run_sft "${CHECKPOINT:-$1}"
        ;;
    sft-test)
        check_environment
        run_sft_test
        ;;
    rl)
        check_environment
        shift
        run_rl "${CHECKPOINT:-$1}"
        ;;
    rl-test)
        check_environment
        run_rl_test
        ;;
    inference)
        check_environment
        shift
        run_inference "${CHECKPOINT:-$1}"
        ;;
    chat)
        check_environment
        shift
        run_chat "${CHECKPOINT:-$1}"
        ;;
    web-chat)
        check_environment
        run_web_chat
        ;;
    tensorboard)
        run_tensorboard
        ;;
    full)
        check_environment
        run_full_pipeline
        ;;
    full-test)
        check_environment
        run_full_pipeline_test
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        print_usage
        exit 1
        ;;
esac
