#!/bin/bash
# Session management script for hashcat Ethereum wallet cracking

SESSION_DIR="/crack-session"

case "${1:-list}" in
    list)
        echo "=========================================="
        echo "Active Hashcat Sessions"
        echo "=========================================="
        
        if [ -d "$SESSION_DIR" ]; then
            cd "$SESSION_DIR"
            
            # Find all .restore files
            restore_files=$(ls *.restore 2>/dev/null)
            
            if [ -z "$restore_files" ]; then
                echo "No active sessions found."
            else
                echo ""
                for file in *.restore; do
                    if [ -f "$file" ]; then
                        session="${file%.restore}"
                        
                        # Parse session name: eth_ADDRESS_sLENGTH_MODE
                        if [[ "$session" =~ ^eth_([a-f0-9]+)_s([0-9]+)_(.+)$ ]]; then
                            address="${BASH_REMATCH[1]}"
                            suffix_len="${BASH_REMATCH[2]}"
                            mode="${BASH_REMATCH[3]}"
                            
                            # Get file size and modification time
                            size=$(du -h "$file" | cut -f1)
                            modified=$(stat -c "%y" "$file" | cut -d' ' -f1,2 | cut -d'.' -f1)
                            
                            echo "Session: $session"
                            echo "  Wallet: 0x${address}"
                            echo "  Suffix: ${suffix_len} characters"
                            echo "  Mode:   ${mode}"
                            echo "  Size:   ${size}"
                            echo "  Modified: ${modified}"
                            echo ""
                        else
                            echo "Session: $session (unknown format)"
                            echo ""
                        fi
                    fi
                done
            fi
        else
            echo "Session directory not found."
        fi
        ;;
        
    clear)
        if [ -z "$2" ]; then
            echo "Usage: $0 clear <session_name|all>"
            echo ""
            echo "Examples:"
            echo "  $0 clear all                                    # Clear all sessions"
            echo "  $0 clear eth_579f2f10d38787ffb573f0ce3370f196f357fa69_s2_mask  # Clear specific session"
            exit 1
        fi
        
        if [ "$2" = "all" ]; then
            echo "Clearing all sessions..."
            if [ -d "$SESSION_DIR" ]; then
                rm -f "$SESSION_DIR"/*.restore "$SESSION_DIR"/*.log "$SESSION_DIR"/*.outfiles 2>/dev/null
                echo "✅ All sessions cleared."
            else
                echo "No session directory found."
            fi
        else
            session_name="$2"
            echo "Clearing session: $session_name"
            if [ -d "$SESSION_DIR" ]; then
                cd "$SESSION_DIR"
                if [ -f "${session_name}.restore" ]; then
                    rm -f "${session_name}.restore" "${session_name}.log" "${session_name}.outfiles" 2>/dev/null
                    echo "✅ Session cleared: $session_name"
                else
                    echo "❌ Session not found: $session_name"
                fi
            else
                echo "No session directory found."
            fi
        fi
        ;;
        
    info)
        if [ -z "$2" ]; then
            echo "Usage: $0 info <session_name>"
            exit 1
        fi
        
        session_name="$2"
        if [ -d "$SESSION_DIR" ] && [ -f "$SESSION_DIR/${session_name}.restore" ]; then
            echo "=========================================="
            echo "Session Details: $session_name"
            echo "=========================================="
            
            # Show file details
            ls -lah "$SESSION_DIR/${session_name}".* 2>/dev/null || true
            
            # Try to extract progress from log if it exists
            if [ -f "$SESSION_DIR/${session_name}.log" ]; then
                echo ""
                echo "Last status from log:"
                tail -20 "$SESSION_DIR/${session_name}.log" | grep -E "(Progress|Speed|Time\.(Started|Estimated))" || echo "No status found in log"
            fi
        else
            echo "Session not found: $session_name"
        fi
        ;;
        
    *)
        echo "Usage: $0 {list|clear|info} [args]"
        echo ""
        echo "Commands:"
        echo "  list              - List all active sessions"
        echo "  clear <name|all>  - Clear specific session or all sessions"
        echo "  info <name>       - Show detailed info about a session"
        exit 1
        ;;
esac