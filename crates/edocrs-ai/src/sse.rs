//! SSE (Server-Sent Events) 字节流切分器 —— 与具体 wire 协议无关的传输层工具。
//!
//! SSE 格式要点 (WHATWG 规范的子集):
//!   - 事件之间以**空行**分隔 (`\n\n`, 也可能是 `\r\n\r\n`);
//!   - 事件内每行 `field: value`, 我们只关心 `data` 字段;
//!   - 一个事件可有多行 `data:`, 值之间用 `\n` 拼接;
//!   - 冒号后的**单个**空格可省略 (`data:x` 与 `data: x` 等价);
//!   - 以 `:` 开头的是注释行 (OpenRouter 用 `: OPENROUTER PROCESSING` 做心跳), 忽略。
//!
//! 学习点: 这是一个最小的「增量协议解码器」—— 网络字节随意断在任何位置 (甚至 UTF-8
//!         多字节字符中间), 所以必须先按字节缓冲、确认拿到完整事件后才转成字符串。

/// 增量 SSE 切分器。喂入任意字节, 吐出已完整的事件的 data payload。
#[derive(Default)]
pub struct SseSplitter {
    /// 尚未凑成完整事件的字节。
    buf: Vec<u8>,
}

impl SseSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// 喂入一段字节, 返回本次能完整切出的 data payload 列表 (无 data 行的事件被丢弃)。
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        // 学习点: `while let Some(..) = ..` 反复找分隔符, 每找到一个就从 buf 头部
        //         `drain` 掉该事件 —— drain 返回被移除元素的迭代器, 同时原地收缩 Vec。
        while let Some((end, sep_len)) = find_event_end(&self.buf) {
            let event: Vec<u8> = self.buf.drain(..end + sep_len).collect();
            if let Some(payload) = extract_data(&event[..end]) {
                out.push(payload);
            }
        }
        out
    }
}

/// 找第一个事件分隔 (空行) 的位置, 返回 (事件字节长度, 分隔符长度)。
///
/// 同时识别 `\n\n` 与 `\r\n\r\n`, 取先出现者。
fn find_event_end(buf: &[u8]) -> Option<(usize, usize)> {
    let lf = buf.windows(2).position(|w| w == b"\n\n").map(|i| (i, 2));
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| (i, 4));
    // 学习点: 两个 Option 取「较早的那个」—— 用 match 元组把四种组合摊开, 比嵌套 if 清楚。
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (a, b) => a.or(b),
    }
}

/// 从一个完整事件里提取 data 值; 多个 data 行用 `\n` 拼接。无 data 行返回 None。
fn extract_data(event: &[u8]) -> Option<String> {
    // 学习点: from_utf8_lossy 遇到非法字节用 U+FFFD 替换而不是失败 —— 流里偶发坏字节
    //         不该让整次采样崩掉。返回 Cow<str>, 合法时零拷贝。
    let text = String::from_utf8_lossy(event);
    let mut data: Option<String> = None;
    for line in text.lines() {
        // `lines()` 会去掉 `\n`, 但 `\r\n` 场景下可能残留 `\r`。
        let line = line.strip_suffix('\r').unwrap_or(line);
        let Some(rest) = line.strip_prefix("data:") else {
            continue; // 注释行 / event: / id: / retry: 一律忽略
        };
        let value = rest.strip_prefix(' ').unwrap_or(rest);
        match data.as_mut() {
            Some(d) => {
                d.push('\n');
                d.push_str(value);
            }
            None => data = Some(value.to_string()),
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_event() {
        let mut s = SseSplitter::new();
        assert_eq!(s.feed(b"data: hello\n\n"), vec!["hello"]);
    }

    /// 事件被网络切成任意碎片, 仍能正确拼回。
    #[test]
    fn event_split_across_chunks() {
        let mut s = SseSplitter::new();
        assert!(s.feed(b"data: hel").is_empty());
        assert!(s.feed(b"lo\n").is_empty());
        assert_eq!(s.feed(b"\n"), vec!["hello"]);
    }

    /// UTF-8 多字节字符被切在中间也不会乱码。
    #[test]
    fn utf8_split_mid_char() {
        let bytes = "data: 你好\n\n".as_bytes();
        let mut s = SseSplitter::new();
        assert!(s.feed(&bytes[..8]).is_empty()); // "你" 的 3 字节被切开
        assert_eq!(s.feed(&bytes[8..]), vec!["你好"]);
    }

    #[test]
    fn crlf_and_no_space_and_comments() {
        let mut s = SseSplitter::new();
        let got = s.feed(b": OPENROUTER PROCESSING\r\n\r\ndata:{\"a\":1}\r\n\r\n");
        assert_eq!(got, vec!["{\"a\":1}"]);
    }

    #[test]
    fn multiple_data_lines_join_with_newline() {
        let mut s = SseSplitter::new();
        assert_eq!(s.feed(b"data: a\ndata: b\n\n"), vec!["a\nb"]);
    }

    #[test]
    fn done_marker_passes_through() {
        let mut s = SseSplitter::new();
        assert_eq!(s.feed(b"data: [DONE]\n\n"), vec!["[DONE]"]);
    }
}
