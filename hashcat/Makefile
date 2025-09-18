# Makefile for Hashcat Ethereum Wallet Cracker

.PHONY: build run crack-1 crack-2 crack-3 crack-4 crack-5 resume clean stop benchmark test

# Build the Docker container
build:
	@echo "Building Hashcat Docker container..."
	docker-compose build

# Run with default settings (1 character suffix)
run: build
	@echo "Running with 1-character suffix..."
	SUFFIX_LENGTH=1 docker-compose up

# Crack with specific suffix lengths
crack-1: build
	@echo "Cracking with 1-character suffix..."
	SUFFIX_LENGTH=1 docker-compose up

crack-2: build
	@echo "Cracking with 2-character suffix..."
	SUFFIX_LENGTH=2 docker-compose up

crack-3: build
	@echo "Cracking with 3-character suffix..."
	SUFFIX_LENGTH=3 docker-compose up

crack-4: build
	@echo "Cracking with 4-character suffix..."
	@echo "WARNING: This will test 95^4 = 81,450,625 combinations per base password!"
	SUFFIX_LENGTH=4 docker-compose up

crack-5: build
	@echo "Cracking with 5-character suffix..."
	@echo "WARNING: This will test 95^5 = 7,737,809,375 combinations per base password!"
	@echo "This could take a very long time even with GPU acceleration!"
	SUFFIX_LENGTH=5 docker-compose up

# Resume previous session
resume:
	@echo "Resuming previous cracking session..."
	RESUME=1 docker-compose up

# Resume specific suffix length session
resume-1:
	@echo "Resuming 1-character suffix session..."
	SUFFIX_LENGTH=1 RESUME=1 docker-compose up

resume-2:
	@echo "Resuming 2-character suffix session..."
	SUFFIX_LENGTH=2 RESUME=1 docker-compose up

resume-3:
	@echo "Resuming 3-character suffix session..."
	SUFFIX_LENGTH=3 RESUME=1 docker-compose up

# Session management
sessions: build
	@echo "Listing all active sessions..."
	docker-compose run --rm hashcat /crack/manage_sessions.sh list

session-info: build
	@echo "Usage: make session-info SESSION=<session_name>"
	@if [ -z "$(SESSION)" ]; then \
		echo "Error: SESSION not specified"; \
		echo "Example: make session-info SESSION=eth_579f2f10d38787ffb573f0ce3370f196f357fa69_s2_mask"; \
		exit 1; \
	fi
	docker-compose run --rm hashcat /crack/manage_sessions.sh info $(SESSION)

clear-session: build
	@echo "Usage: make clear-session SESSION=<session_name>"
	@if [ -z "$(SESSION)" ]; then \
		echo "Error: SESSION not specified"; \
		echo "Example: make clear-session SESSION=eth_579f2f10d38787ffb573f0ce3370f196f357fa69_s2_mask"; \
		echo "Or: make clear-session SESSION=all"; \
		exit 1; \
	fi
	docker-compose run --rm hashcat /crack/manage_sessions.sh clear $(SESSION)

clear-all-sessions: build
	@echo "Clearing all sessions..."
	docker-compose run --rm hashcat /crack/manage_sessions.sh clear all

# Run benchmark
benchmark: build
	@echo "Running hashcat benchmark for Ethereum modes..."
	docker-compose run --rm hashcat hashcat -b -m 15700
	docker-compose run --rm hashcat hashcat -b -m 15600

# Test with known password
test: build
	@echo "Testing with known password k0sk3sh\$$12..."
	@echo "k0sk3sh\$$1" > test-passwords.txt
	docker-compose run --rm \
		-v $$(PWD)/test-passwords.txt:/data/passwords.txt \
		hashcat bash -c "SUFFIX_LENGTH=1 /crack/crack.sh"

# Clean up
clean:
	@echo "Cleaning up..."
	docker-compose down
	rm -rf crack-session/* results/*
	rm -f test-passwords.txt

# Stop running container
stop:
	@echo "Stopping hashcat container..."
	docker-compose down

# Show GPU info
gpu-info: build
	@echo "GPU Information:"
	docker-compose run --rm hashcat hashcat -I

# Interactive shell for debugging
shell: build
	@echo "Starting interactive shell..."
	docker-compose run --rm hashcat bash

# Monitor progress (if running)
monitor:
	@echo "Monitoring hashcat progress..."
	docker logs -f hashcat-cracker

# Quick test with wordlist mode (for small suffix)
wordlist-1: build
	@echo "Using wordlist mode with 1-character suffix..."
	SUFFIX_LENGTH=1 ATTACK_MODE=wordlist docker-compose up

wordlist-2: build
	@echo "Using wordlist mode with 2-character suffix..."
	SUFFIX_LENGTH=2 ATTACK_MODE=wordlist docker-compose up

# Help
help:
	@echo "Hashcat Ethereum Wallet Cracker"
	@echo "================================"
	@echo ""
	@echo "Basic Usage:"
	@echo "  make build        - Build the Docker container"
	@echo "  make crack-1      - Crack with 1-character suffix"
	@echo "  make crack-2      - Crack with 2-character suffix"
	@echo "  make crack-3      - Crack with 3-character suffix"
	@echo "  make crack-4      - Crack with 4-character suffix (slow!)"
	@echo "  make crack-5      - Crack with 5-character suffix (very slow!)"
	@echo ""
	@echo "Session Management:"
	@echo "  make resume-1     - Resume 1-character suffix session"
	@echo "  make resume-2     - Resume 2-character suffix session"
	@echo "  make resume-3     - Resume 3-character suffix session"
	@echo "  make sessions     - List all active sessions"
	@echo "  make session-info SESSION=<name> - Show session details"
	@echo "  make clear-session SESSION=<name> - Clear specific session"
	@echo "  make clear-all-sessions - Clear all sessions"
	@echo ""
	@echo "Utilities:"
	@echo "  make benchmark    - Run performance benchmark"
	@echo "  make test         - Test with known password"
	@echo "  make gpu-info     - Show GPU information"
	@echo "  make monitor      - Monitor running session"
	@echo "  make stop         - Stop running container"
	@echo "  make clean        - Clean up files"
	@echo "  make shell        - Interactive shell for debugging"
	@echo ""
	@echo "Advanced:"
	@echo "  make wordlist-1   - Use wordlist mode for 1-char suffix"
	@echo "  make wordlist-2   - Use wordlist mode for 2-char suffix"
	@echo ""
	@echo "The system creates unique sessions for each wallet+suffix combination."
	@echo "Sessions persist across container restarts for automatic resume."