#include <iostream>
#include <string>

int main() {
    std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?/~`";
    std::cout << "Charset length: " << charset.length() << std::endl;
    std::cout << "Characters:" << std::endl;
    
    for (size_t i = 0; i < charset.length(); i++) {
        std::cout << i << ": '" << charset[i] << "'";
        if (charset[i] >= '0' && charset[i] <= '9') {
            std::cout << " (digit)";
        }
        std::cout << std::endl;
    }
    
    // Show what character 5 would be (for k0sk3sh$1f)
    std::cout << "\nCharacter at index 5: '" << charset[5] << "'" << std::endl;
    
    // Show characters 52-61 (where digits are)
    std::cout << "\nDigits are at indices:" << std::endl;
    for (int i = 52; i < 62; i++) {
        std::cout << i << ": '" << charset[i] << "'" << std::endl;
    }
    
    return 0;
}