#ifndef REGISTRATION_CSV_PARSER_H
#define REGISTRATION_CSV_PARSER_H

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class CSVRow {
public:
    std::string operator[](std::size_t index) const;
    std::size_t size() const;
    void readNextRow(std::istream& str);
private:
    std::string         m_line;
    std::vector<int>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data);
#endif
