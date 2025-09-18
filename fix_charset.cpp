#include <iostream>
#include <string>

int main() {
    // Original charset (91 chars)
    std::string charset1 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?/~`";
    
    // Full printable ASCII (94 chars) - all printable chars from space to tilde
    std::string charset2;
    for (char c = '!'; c <= '~'; c++) {
        if (c != '"' && c != '\\' && c != '\'') { // Skip quotes and backslash for C string compatibility
            charset2 += c;
        }
    }
    
    std::cout << "Original charset length: " << charset1.length() << std::endl;
    std::cout << "Full charset length: " << charset2.length() << std::endl;
    
    // Show missing characters
    std::cout << "\nCharacters in full but not in original:" << std::endl;
    for (char c : charset2) {
        if (charset1.find(c) == std::string::npos) {
            std::cout << "  '" << c << "' (ASCII " << (int)c << ")" << std::endl;
        }
    }
    
    // Fixed charset with exactly 94 chars
    std::string fixed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;':,.<>?/~`\" \\";
    std::cout << "\nFixed charset (94 chars): length=" << fixed.length() << std::endl;
    
    return 0;
}