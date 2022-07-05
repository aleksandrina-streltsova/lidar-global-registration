#include <fstream>

#include "utils.h"

int debugPoint(int line)
{
    if (line < 0)
        return 0;

    // You can put breakpoint at the following line to catch any rassert failure:
    return line;
}

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

void saveVector(const std::vector<float> &vs, const std::string &filepath) {
    std::fstream fout(filepath, std::ios_base::out);
    if (!fout.is_open())
        perror(("error while opening file " + filepath).c_str());

    fout << "value\n";
    for (float v: vs) {
        fout << v << "\n";
    }
    fout.close();
}
