#ifndef WALLET_H
#define WALLET_H

#include "types.h"
#include <json/json.h>

class WalletLoader {
public:
    static bool loadFromFile(const std::string& filepath, WalletData& wallet);
    static bool validatePassword(const WalletData& wallet, const std::string& password,
                                 std::vector<uint8_t>& private_key);
    
private:
    static bool parseKeystoreV3(const Json::Value& root, WalletData& wallet);
    static bool deriveKey(const std::string& password, const WalletData& wallet,
                          std::vector<uint8_t>& derived_key);
};

#endif