#include <iostream>
#include <string>

int main() {
    std::string base = "k0sk3sh$1";
    std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?/~`";
    
    std::cout << "Testing password: " << base << " + 1 character" << std::endl;
    std::cout << "Total: " << charset.length() << " combinations" << std::endl;
    std::cout << "\nPasswords to test:" << std::endl;
    
    for (size_t i = 0; i < charset.length(); i++) {
        std::cout << "  " << (i+1) << ". " << base << charset[i] << std::endl;
    }
    
    return 0;
}