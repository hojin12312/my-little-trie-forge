#pragma once

#include <string>
#include <string_view>

namespace splash::json {

[[nodiscard]] inline std::string quote(std::string_view value) {
  std::string result;
  result.reserve(value.size() + 2);
  result.push_back('"');
  for (unsigned char character : value) {
    switch (character) {
    case '"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\b':
      result += "\\b";
      break;
    case '\f':
      result += "\\f";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      if (character < 0x20) {
        constexpr char hex[] = "0123456789abcdef";
        result += "\\u00";
        result.push_back(hex[character >> 4]);
        result.push_back(hex[character & 0x0f]);
      } else {
        result.push_back(static_cast<char>(character));
      }
    }
  }
  result.push_back('"');
  return result;
}

} // namespace splash::json
