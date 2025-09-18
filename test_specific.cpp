#include <iostream>
#include <string>

int main() {
    std::string charset = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
    std::string base = "k0sk3sh$1";
    
    std::cout << "Charset length: " << charset.length() << std::endl;
    std::cout << "Looking for password: k0sk3sh$12" << std::endl;
    
    // Find position of '2'
    for (size_t i = 0; i < charset.length(); i++) {
        if (charset[i] == '2') {
            std::cout << "Character '2' is at position " << i << std::endl;
            std::cout << "Password would be: " << base << charset[i] << std::endl;
            std::cout << "This should be in batch 1 (positions 0-31)" << std::endl;
        }
    }
    
    // Show what's being tested at position 18
    std::cout << "\nAt position 18: '" << charset[18] << "'" << std::endl;
    std::cout << "Password at position 18: " << base << charset[18] << std::endl;
    
    // Show all digits
    std::cout << "\nAll digit positions:" << std::endl;
    for (size_t i = 0; i < charset.length(); i++) {
        if (charset[i] >= '0' && charset[i] <= '9') {
            std::cout << "  Position " << i << ": '" << charset[i] << "' -> " << base << charset[i] << std::endl;
        }
    }
    
    return 0;
}