"""
core/zh_codec.py — 中文↔ASCII 序列编码译码器 (v14.5)

设计哲学:
  - 生物脉冲神经网络核心 (lif_pytorch.py) 保持纯 8-bit ASCII 不变
  - 中文在进入/离开网络时进行编解码 — 与网络架构完全解耦
  - 无 vocab 表、无 13-bit 字库、无输出层改动
  - 红线合规: 编解码只是单向确定映射, 不含任何学习/数值信号

编码方案:
  - ASCII 可打印字符 (32-126): 直接通过
  - 反斜杠 `\\` (92): 转义为 `\\` (两个反斜杠, 即 2 个 ASCII 92)
  - 非 ASCII 字符: `\\uXXXX` 格式 (Unicode 16 进制码点, 大写字)
    例如: 中 → \\u4E2D, 国 → \\u56FD, 你 → \\u4F60
    占 6 个 ASCII 字符 (1 反斜杠 + 1 u + 4 hex)

  确定性: 同一字符永远编码为同一 ASCII 序列。
  可逆性: 无歧义 — 先匹配 `\\\\` (转义反斜杠), 再匹配 `\\uXXXX` (Unicode 转义)。
"""

import re

# 编译正则: 匹配 `\\uXXXX` (1 反斜杠 + u + 4 hex)
_U_ESCAPE_RE = re.compile(r"\\u([0-9A-Fa-f]{4})")


def encode(text):
    """中文文本 → 纯 ASCII 字符序列 (网络直接处理)

    Args:
        text: 原始文本 (可含中文/任意 Unicode)

    Returns:
        ascii_text: 纯 ASCII 字符串, 仅含 32-126 可打印字符
    """
    result = []
    for ch in text:
        code = ord(ch)
        if code == 92:  # 反斜杠本身需要转义
            result.append("\\\\")
        elif 32 <= code <= 126:
            result.append(ch)
        else:
            result.append("\\u{:04X}".format(code))
    return "".join(result)


def decode(text):
    """ASCII 字符序列 → 还原中文文本

    Args:
        text: 网络生成的纯 ASCII 字符串

    Returns:
        original_text: 还原后的文本 (含中文)
    """
    result = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        # 先检查转义反斜杠 `\\` → `\`
        if ch == "\\" and i + 1 < n and text[i + 1] == "\\":
            result.append("\\")
            i += 2
        # 再检查 Unicode 转义 `\uXXXX` → 中文字符
        elif ch == "\\" and i + 5 < n and text[i + 1] == "u":
            hex_str = text[i + 2:i + 6]
            try:
                result.append(chr(int(hex_str, 16)))
                i += 6
            except (ValueError, OverflowError):
                result.append(ch)
                i += 1
        else:
            result.append(ch)
            i += 1
    return "".join(result)


def count_chars(text):
    """统计"中文视角"的字符数

    网络生成 n 个 ASCII 字符, 还原后可能是少于 n 个中文字。
    返回值: (还原后字符数, 编码后 ASCII 字符数)
    """
    raw = encode(text)
    return len(decode(raw)), len(raw)


def is_ascii_only(text):
    """检查文本是否纯 ASCII"""
    return all(32 <= ord(c) <= 126 for c in text)