.PHONY: help build run crack resume clean stop logs monitor setup test benchmark docker-build docker-up docker-down

# Default target
help:
	@echo "CryptoCracker - GPU-Accelerated Ethereum Wallet Recovery"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup          - Install dependencies and setup environment"
	@echo "  make build          - Build the application locally"
	@echo "  make docker-build   - Build Docker container with CUDA support"
	@echo "  make docker-up      - Start all services with docker-compose"
	@echo "  make docker-down    - Stop all services"
	@echo "  make crack          - Start cracking with default settings"
	@echo "  make resume         - Resume from last checkpoint"
	@echo "  make monitor        - Open monitoring dashboard"
	@echo "  make logs           - Show container logs"
	@echo "  make clean          - Clean build artifacts and checkpoints"
	@echo "  make test           - Run tests"
	@echo "  make benchmark      - Run performance benchmark"

# Setup environment
setup:
	@echo "Setting up environment..."
	mkdir -p wallets checkpoints logs config monitoring/dashboards monitoring/datasources
	@echo "Checking CUDA installation..."
	@nvidia-smi || (echo "Error: NVIDIA drivers not found" && exit 1)
	@echo "Setup complete!"

# Build locally
build:
	@echo "Building CryptoCracker..."
	mkdir -p build
	cd build && cmake .. && make -j$$(nproc)
	@echo "Build complete! Binary at: build/cryptocracker"

# Docker commands
docker-build:
	@echo "Building Docker container with CUDA support..."
	docker-compose build
	@echo "Container built successfully!"

docker-up:
	@echo "Starting CryptoCracker services..."
	docker-compose -f docker-compose-simple.yml up -d
	@echo "Services started!"

docker-down:
	@echo "Stopping services..."
	docker-compose -f docker-compose-simple.yml down
	@echo "Services stopped."

# Crack wallet with configuration
crack: docker-up
	@echo "Starting wallet cracking..."
	@if [ ! -f "wallets/wallet.json" ]; then \
		echo "Error: Place your wallet file at wallets/wallet.json"; \
		exit 1; \
	fi
	@if [ ! -f "config/passwords.txt" ]; then \
		echo "Error: Place base passwords at config/passwords.txt"; \
		exit 1; \
	fi
	docker exec -it cryptocracker ./build/cryptocracker \
		--wallet /app/wallets/wallet.json \
		--passwords /app/config/passwords.txt \
		--gpu 0 \
		--checkpoint /app/checkpoints/session.db \
		--log /app/logs/crack.log

# Resume from checkpoint
resume: docker-up
	@echo "Resuming from checkpoint..."
	docker exec -it cryptocracker ./build/cryptocracker \
		--resume \
		--checkpoint /app/checkpoints/session.db \
		--log /app/logs/crack.log

# Quick crack for testing (limited search)
quick-crack: docker-up
	@echo "Running quick crack test..."
	docker exec -it cryptocracker ./build/cryptocracker \
		--wallet /app/wallets/wallet.json \
		--passwords /app/config/passwords.txt \
		--gpu 0 \
		--max-attempts 1000000 \
		--suffix-length 4

# Monitor performance
monitor:
	@echo "Opening monitoring dashboard..."
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@xdg-open http://localhost:3000 2>/dev/null || open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000 manually"

# View logs
logs:
	docker-compose logs -f cryptocracker

logs-all:
	docker-compose logs -f

# Clean artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	@echo "Clean complete. Checkpoints preserved."

clean-all: clean
	@echo "Cleaning all data including checkpoints..."
	rm -rf checkpoints/* logs/*
	docker-compose down -v
	@echo "Full clean complete."

# Testing
test: docker-build
	@echo "Running tests..."
	docker run --rm --gpus all cryptocracker:latest ./build/cryptocracker --test

# Benchmark GPU performance
benchmark: docker-build
	@echo "Running GPU benchmark..."
	docker run --rm --gpus all cryptocracker:latest ./build/cryptocracker --benchmark

# Development mode with live reload
dev:
	@echo "Starting in development mode..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Check GPU status
gpu-status:
	@echo "GPU Status:"
	@nvidia-smi
	@echo ""
	@echo "Docker GPU access:"
	@docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Install dependencies (for local development)
install-deps:
	@echo "Installing system dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential cmake libssl-dev libsqlite3-dev libjsoncpp-dev libscrypt-dev
	@echo "Dependencies installed!"

# Format code
format:
	find src include -name "*.cpp" -o -name "*.cu" -o -name "*.h" | xargs clang-format -i

# Static analysis
analyze:
	cppcheck --enable=all --suppress=missingIncludeSystem src/ include/

# Create sample configuration
init-config:
	@echo "Creating sample configuration files..."
	@echo "password1" > config/passwords.txt
	@echo "password2" >> config/passwords.txt
	@echo "mypassword" >> config/passwords.txt
	@echo "Sample passwords created at config/passwords.txt"
	@echo ""
	@echo "Place your wallet file at: wallets/wallet.json"
	@echo "Your wallet filename: UTC--2016-05-01T18-11-51.988445605Z--579f2f10d38787ffb573f0ce3370f196f357fa69"