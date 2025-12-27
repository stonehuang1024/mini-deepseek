#!/bin/bash
# DeepSeek V3 Pretraining Script
# ================================
# This script provides convenient ways to run pretraining with different configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to print usage
print_usage() {
    echo -e "${BLUE}DeepSeek V3 Pretraining${NC}"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  quick-test    Run a quick test with minimal data (50 steps)"
    echo "  small         Train on WikiText-2 dataset (~13MB, ~500 steps)"
    echo "  medium        Train on WikiText-103 dataset (~500MB, ~5000 steps)"
    echo "  large         Train on OpenWebText subset (~10GB, ~50000 steps)"
    echo "  full          Train on full OpenWebText (~40GB)"
    echo "  tensorboard   Start TensorBoard server"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 quick-test    # Quick sanity check"
    echo "  $0 small         # Small scale training"
    echo "  $0 tensorboard   # Monitor training"
}

# Function to update config for different scales
update_config_for_scale() {
    local scale=$1
    local config_file="config_default.yaml"
    
    case $scale in
        quick-test)
            echo -e "${YELLOW}Setting up quick test configuration...${NC}"
            # Use sed to update config (keeping wikitext for quick test)
            sed -i.bak 's/dataset_name: openwebtext/dataset_name: wikitext/' "$config_file"
            sed -i.bak 's/dataset_config: null/dataset_config: wikitext-2-raw-v1/' "$config_file"
            sed -i.bak 's/max_steps: [0-9]*/max_steps: 50/' "$config_file"
            ;;
        small)
            echo -e "${YELLOW}Setting up small scale configuration (WikiText-2)...${NC}"
            sed -i.bak 's/dataset_name: openwebtext/dataset_name: wikitext/' "$config_file"
            sed -i.bak 's/dataset_config: null/dataset_config: wikitext-2-raw-v1/' "$config_file"
            sed -i.bak 's/max_steps: [0-9]*/max_steps: 500/' "$config_file"
            ;;
        medium)
            echo -e "${YELLOW}Setting up medium scale configuration (WikiText-103)...${NC}"
            sed -i.bak 's/dataset_name: wikitext/dataset_name: wikitext/' "$config_file"
            sed -i.bak 's/dataset_config: wikitext-2-raw-v1/dataset_config: wikitext-103-raw-v1/' "$config_file"
            sed -i.bak 's/max_steps: [0-9]*/max_steps: 5000/' "$config_file"
            ;;
        large)
            echo -e "${YELLOW}Setting up large scale configuration (OpenWebText subset ~10GB)...${NC}"
            sed -i.bak 's/dataset_name: wikitext/dataset_name: openwebtext/' "$config_file"
            sed -i.bak 's/dataset_config: wikitext-[0-9]*-raw-v1/dataset_config: null/' "$config_file"
            sed -i.bak 's/max_samples: null/max_samples: 2000000/' "$config_file"
            sed -i.bak 's/max_steps: [0-9]*/max_steps: 50000/' "$config_file"
            ;;
        full)
            echo -e "${YELLOW}Setting up full scale configuration (Full OpenWebText ~40GB)...${NC}"
            sed -i.bak 's/dataset_name: wikitext/dataset_name: openwebtext/' "$config_file"
            sed -i.bak 's/dataset_config: wikitext-[0-9]*-raw-v1/dataset_config: null/' "$config_file"
            sed -i.bak 's/max_samples: [0-9]*/max_samples: null/' "$config_file"
            sed -i.bak 's/max_steps: [0-9]*/max_steps: 200000/' "$config_file"
            ;;
    esac
    
    # Clean up backup file
    rm -f "${config_file}.bak"
}

# Main script logic
case "${1:-help}" in
    quick-test)
        echo -e "${GREEN}Starting quick test training...${NC}"
        python train.py --mode pretrain --test
        ;;
    small)
        echo -e "${GREEN}Starting small scale training (WikiText-2)...${NC}"
        update_config_for_scale small
        python train.py --mode pretrain
        ;;
    medium)
        echo -e "${GREEN}Starting medium scale training (WikiText-103)...${NC}"
        update_config_for_scale medium
        python train.py --mode pretrain
        ;;
    large)
        echo -e "${GREEN}Starting large scale training (OpenWebText ~10GB)...${NC}"
        echo -e "${YELLOW}Warning: This will download ~10GB of data!${NC}"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            update_config_for_scale large
            python train.py --mode pretrain
        fi
        ;;
    full)
        echo -e "${GREEN}Starting full scale training (Full OpenWebText ~40GB)...${NC}"
        echo -e "${YELLOW}Warning: This will download ~40GB of data!${NC}"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            update_config_for_scale full
            python train.py --mode pretrain
        fi
        ;;
    tensorboard)
        echo -e "${GREEN}Starting TensorBoard...${NC}"
        echo -e "${BLUE}Open http://localhost:6006 in your browser${NC}"
        tensorboard --logdir=runs --port=6006
        ;;
    help|*)
        print_usage
        ;;
esac
