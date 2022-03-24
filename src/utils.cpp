#include "utils.h"

void split(const std::string &str, std::vector<std::string> &tokens, const std::string &delimiter) {
    tokens.clear();
    std::string s(str);
    std::size_t pos = 0;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        tokens.push_back(s.substr(0, pos));
        s.erase(0, pos + delimiter.length());
    }
    if (!s.empty()) {
        tokens.push_back(s);
    }
}

