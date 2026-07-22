use std::io::Write;

use jsonc_bytecode::{Address, Literal, OpCode};
use jsonc_mlir::debug;

const OPCODE_LOAD_LOCAL_INLINE_BASE: u8 = 0x20; // 0x10..=0x1F
const OPCODE_STORE_LOCAL_INLINE_BASE: u8 = 0x30; // 0x20..=0x2F
// const OPCODE_LOAD_GLOBAL_INLINE_BASE: u8 = 0x40; // 0x30..=0x3F
const OPCODE_STORE_GLOBAL_INLINE_BASE: u8 = 0x50; // 0x40..=0x4F
const OPCODE_MAKE_TRUE: u8 = 0x60;
const OPCODE_MAKE_FALSE: u8 = 0x61;
const OPCODE_MAKE_INTEGER_IMM: u8 = 0x62;

/// This function checks if the given string is a numeric string.
/// Numeric strings are strings that represent a valid integer value.
///
/// Returns `Some` with the parsed integer value if the string is a numeric string, otherwise `None`.
fn is_numeric_string(s: &str) -> Option<u64> {
    if s.is_empty() || s.len() > 20 {
        return None;
    }
    if !s.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if let Ok(val) = s.parse::<u64>() {
        if val.to_string() == s {
            return Some(val);
        }
    }
    None
}

/// This function checks if the given string is a timestamp string.
/// Timestamp strings are strings that represent a valid timestamp value.
///
/// It may be in string format `YYYY-MM-DDTHH:MM:SS.mmm+HH:MM`.
///
/// Returns `Some` with the parsed timestamp value if the string is a timestamp string, otherwise `None`.
fn is_timestamp_string(s: &str) -> Option<u64> {
    if s.len() != 32 {
        return None;
    }
    let bytes = s.as_bytes();
    if bytes[4] != b'-' || bytes[7] != b'-' || bytes[10] != b'T' {
        return None;
    }
    if bytes[13] != b':' || bytes[16] != b':' || bytes[19] != b'.' {
        return None;
    }
    if bytes[26] != b'+' || bytes[29] != b':' {
        return None;
    }
    if &s[27..32] != "00:00" {
        return None;
    }
    let parse_digits = |start: usize, end: usize| -> Option<u64> {
        let mut val = 0u64;
        for i in start..end {
            let digit = bytes[i];
            if !digit.is_ascii_digit() {
                return None;
            }
            val = val * 10 + (digit - b'0') as u64;
        }
        Some(val)
    };

    let year = parse_digits(0, 4)?;
    let month = parse_digits(5, 7)?;
    let day = parse_digits(8, 10)?;
    let hour = parse_digits(11, 13)?;
    let minute = parse_digits(14, 16)?;
    let second = parse_digits(17, 19)?;
    let microseconds = parse_digits(20, 26)?;

    if year < 2000 || year > 2255 {
        return None;
    }
    if month < 1 || month > 12 {
        return None;
    }
    if day < 1 || day > 31 {
        return None;
    }
    if hour > 23 || minute > 59 || second > 59 || microseconds > 999999 {
        return None;
    }

    let packed: u64 = (year - 2000) << 46
        | month << 42
        | day << 37
        | hour << 32
        | minute << 26
        | second << 20
        | microseconds;

    Some(packed)
}

/// Encode unsigned LEB128
/// This encodes a arbitrary unsigned 64-bit integer into a LEB128-encoded byte sequence.
/// This permits encoding any positive integer up to 2^64 - 1 using a variable-length
/// encoding.
fn encode_uleb128_into(mut mut_val: u64, buf: &mut Vec<u8>) {
    loop {
        let mut byte = (mut_val & 0x7F) as u8;
        mut_val >>= 7;
        if mut_val != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if mut_val == 0 {
            break;
        }
    }
}

/// Encode signed LEB128 (SLEB128)
/// This encodes a arbitrary signed 64-bit integer into a LEB128-encoded byte sequence.
/// This permits encoding any integer up to 2^63 - 1 using a variable-length
/// encoding.
fn encode_sleb128_into(mut mut_val: i64, buf: &mut Vec<u8>) {
    loop {
        let byte = (mut_val & 0x7F) as u8;
        let sign_bit = (byte & 0x40) != 0;
        mut_val >>= 7; // arithmetic shift in Rust for signed
        let done = (mut_val == 0 && !sign_bit) || (mut_val == -1 && sign_bit);
        let mut out = byte;
        if !done {
            out |= 0x80;
        }
        buf.push(out);
        if done {
            break;
        }
    }
}

/// This function encodes an instruction into the given buffer.
/// It uses the `OpCode` enum to determine the instruction to encode.
/// The encoded instruction is appended to the buffer.
fn encode_into(instr: &OpCode, buf: &mut Vec<u8>) {
    /// Encodes an inline index into the buffer.
    /// If the index is less than 16, it is encoded as an inline byte; otherwise,
    /// it is encoded as a fallback opcode followed by an ULEB128-encoded index.
    fn encode_inline_index(
        inline_base: u8,
        fallback_opcode: u8,
        index: Address,
        buf: &mut Vec<u8>,
    ) {
        if index < 16 {
            buf.push(inline_base + index as u8);
        } else {
            buf.push(fallback_opcode);
            encode_uleb128_into(index as u64, buf);
        }
    }

    match instr {
        OpCode::MakeFunction {
            num_params,
            body_len,
        } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*num_params as u64, buf);
            encode_uleb128_into(*body_len as u64, buf);
        }
        OpCode::MakeField { field_name } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*field_name as u64, buf);
        }
        OpCode::MakeObject { num_fields } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*num_fields as u64, buf);
        }
        OpCode::MakeArray { num_elements } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*num_elements as u64, buf);
        }
        OpCode::MakeInteger { value } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*value as u64, buf);
        }
        OpCode::MakeIntegerImm { value } => {
            buf.push(OPCODE_MAKE_INTEGER_IMM);
            encode_sleb128_into(*value, buf);
        }
        OpCode::MakeString { value } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*value as u64, buf);
        }
        OpCode::MakeStringInline { value } => {
            buf.push(instr.opcode());
            encode_uleb128_into(value.len() as u64, buf);
            buf.extend_from_slice(value.as_bytes());
        }
        OpCode::MakeStringNumInline { value } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*value, buf);
        }
        OpCode::MakeStringTsInline { value } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*value, buf);
        }
        OpCode::MakeNull => {
            buf.push(instr.opcode());
        }
        OpCode::MakeBoolean { value } => {
            buf.push(if *value {
                OPCODE_MAKE_TRUE
            } else {
                OPCODE_MAKE_FALSE
            });
        }
        OpCode::CallFunction {
            num_args,
            func_index,
        } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*num_args as u64, buf);
            encode_uleb128_into(*func_index as u64, buf);
        }
        OpCode::StoreGlobal { var_index } => {
            encode_inline_index(
                OPCODE_STORE_GLOBAL_INLINE_BASE,
                instr.opcode(),
                *var_index,
                buf,
            );
        }
        OpCode::StoreLocal { var_index } => {
            encode_inline_index(
                OPCODE_STORE_LOCAL_INLINE_BASE,
                instr.opcode(),
                *var_index,
                buf,
            );
        }
        OpCode::Add => {
            buf.push(instr.opcode());
        }
        OpCode::LoadGlobal { var_index } => {
            if *var_index < 144 {
                buf.push(0x70 + *var_index as u8);
            } else {
                buf.push(12);
                encode_uleb128_into(*var_index as u64, buf);
            }
        }
        OpCode::LoadLocal { var_index } => {
            encode_inline_index(
                OPCODE_LOAD_LOCAL_INLINE_BASE,
                instr.opcode(),
                *var_index,
                buf,
            );
        }
        OpCode::Nop => {
            buf.push(instr.opcode());
        }
        OpCode::MakeFloat { value } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*value as u64, buf);
        }
        OpCode::AddAddr { addr } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*addr as u64, buf);
        }
        OpCode::AddAddr2 { addr1, addr2 } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*addr1 as u64, buf);
            encode_uleb128_into(*addr2 as u64, buf);
        }

        OpCode::MakeFieldFromAddr { field_name, addr } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*field_name as u64, buf);
            encode_uleb128_into(*addr as u64, buf);
        }
        OpCode::AddGlobalLeft { addr } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*addr as u64, buf);
        }
        OpCode::AddGlobalRight { addr } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*addr as u64, buf);
        }
        OpCode::AddGlobal2 { addr1, addr2 } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*addr1 as u64, buf);
            encode_uleb128_into(*addr2 as u64, buf);
        }
        OpCode::MakeFieldFromGlobal { field_name, addr } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*field_name as u64, buf);
            encode_uleb128_into(*addr as u64, buf);
        }
        OpCode::MakeArray2 { addr1, addr2 } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*addr1 as u64, buf);
            encode_uleb128_into(*addr2 as u64, buf);
        }
        OpCode::MakeFieldFromIntegerImm { field_name, value } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*field_name as u64, buf);
            encode_sleb128_into(*value, buf);
        }
        OpCode::MakeFieldFromBoolean { field_name, value } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*field_name as u64, buf);
            buf.push(if *value { 1 } else { 0 });
        }
        OpCode::MakePairArray { pairs } => {
            buf.push(instr.opcode());
            encode_uleb128_into(pairs.len() as u64, buf);
            for (addr1, addr2) in pairs {
                encode_uleb128_into(*addr1 as u64, buf);
                encode_uleb128_into(*addr2 as u64, buf);
            }
        }
        OpCode::MakeFieldIntegerBlock { fields } => {
            buf.push(instr.opcode());
            encode_uleb128_into(fields.len() as u64, buf);
            for (field_name, value) in fields {
                encode_uleb128_into(*field_name as u64, buf);
                encode_sleb128_into(*value, buf);
            }
        }
        OpCode::MakeFieldIntegerBlockSequential {
            count,
            start_value,
            start_field_address,
        } => {
            buf.push(instr.opcode());
            encode_uleb128_into(*count as u64, buf);
            encode_sleb128_into(*start_value, buf);
            encode_uleb128_into(*start_field_address as u64, buf);
        }
        OpCode::MakeFieldGlobalBlock { fields } => {
            buf.push(instr.opcode());
            encode_uleb128_into(fields.len() as u64, buf);
            for (field_name, addr) in fields {
                encode_uleb128_into(*field_name as u64, buf);
                encode_uleb128_into(*addr as u64, buf);
            }
        }
        OpCode::MakeFieldAddrBlock { fields } => {
            buf.push(instr.opcode());
            encode_uleb128_into(fields.len() as u64, buf);
            for (field_name, addr) in fields {
                encode_uleb128_into(*field_name as u64, buf);
                encode_uleb128_into(*addr as u64, buf);
            }
        }
        OpCode::MakeFieldBooleanBlock { fields } => {
            buf.push(instr.opcode());
            encode_uleb128_into(fields.len() as u64, buf);
            for (field_name, value) in fields {
                encode_uleb128_into(*field_name as u64, buf);
                buf.push(if *value { 1 } else { 0 });
            }
        }
        OpCode::MakeFieldFromLocal { field_name, addr } => {
            buf.push(instr.opcode());

            encode_uleb128_into(*field_name as u64, buf);
            encode_uleb128_into(*addr as u64, buf);
        }
        OpCode::MakeFieldLocalBlock { fields } => {
            buf.push(instr.opcode());
            encode_uleb128_into(fields.len() as u64, buf);
            for (field_name, addr) in fields {
                encode_uleb128_into(*field_name as u64, buf);
                encode_uleb128_into(*addr as u64, buf);
            }
        }
    }
}

/// This function encodes a sequence of instructions into the given buffer.
/// It first encodes the number of instructions, then encodes each instruction in turn.
pub fn encode(instr: &OpCode) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8);
    encode_into(instr, &mut buf);
    buf
}

/// This function encodes a sequence of instructions into the given buffer.
/// It first encodes the number of instructions, then encodes each instruction in turn.
fn encode_instrs_into(instrs: &[OpCode], buf: &mut Vec<u8>) {
    encode_uleb128_into(instrs.len() as u64, buf);

    for instr in instrs {
        encode_into(instr, buf);
    }
}

/// This function encodes a sequence of instructions into a byte vector.
/// It first encodes the number of instructions, then encodes each instruction in turn.
pub fn encode_instrs(instrs: &[OpCode]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(instrs.len().saturating_mul(4).saturating_add(8));
    encode_instrs_into(instrs, &mut buf);
    buf
}

/// This function encodes a literal value into the given buffer.
fn encode_literal_into(literal: &Literal, buf: &mut Vec<u8>) {
    match literal {
        // Integer literals are encoded as a tag byte followed by a signed LEB128 value.
        Literal::Integer(i) => {
            buf.push(0x00);
            encode_sleb128_into(*i, buf);
        }
        // String literals are encoded as a tag byte followed by a ULEB128 length
        // and the string bytes themselves.
        Literal::String(s) => {
            if let Some(val) = is_numeric_string(s) {
                buf.push(0x05);
                encode_uleb128_into(val, buf);
            } else if let Some(val) = is_timestamp_string(s) {
                buf.push(0x06);
                encode_uleb128_into(val, buf);
            } else {
                let len = s.len();
                if len < 16 {
                    buf.push(0x10 + len as u8);
                } else {
                    buf.push(0x01);
                    encode_uleb128_into(len as u64, buf);
                }
                buf.extend(s.as_bytes());
            }
        }
        // Float literals are encoded as a tag byte followed by the float's little-endian bytes.
        Literal::Float(f) => {
            buf.push(0x02);
            buf.extend(&f.to_le_bytes());
        }
        // Null literals are encoded as a single byte tag.
        Literal::Null => {
            buf.push(0x03);
        }
        // Boolean literals are encoded as a single byte tag followed by the boolean value.
        Literal::Bool(b) => {
            buf.push(0x04);
            buf.push(*b as u8);
        }
    }
}

/// Encodes a [`Literal`] into a JSONC literal byte sequence.
/// Same behavior as [`encode`].
pub fn encode_literal(literal: &Literal) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8);
    encode_literal_into(literal, &mut buf);
    buf
}

/// Encodes a slice of [`Literal`]s into a JSONC literals byte sequence.
/// Same behavior as [`encode_instrs_into`], but returns a [`Vec`] instead of writing to a buffer.
fn encode_literals_into(literals: &[Literal], buf: &mut Vec<u8>) {
    // Separate strings and non-strings for encoding.
    let mut strings = Vec::new();
    let mut non_strings = Vec::new();
    for lit in literals {
        match lit {
            Literal::String(s) => {
                if is_numeric_string(s).is_some() || is_timestamp_string(s).is_some() {
                    non_strings.push(lit);
                } else {
                    strings.push(s);
                }
            }
            other => non_strings.push(other),
        }
    }

    // Encode non-string literals first, then strings.
    encode_uleb128_into(non_strings.len() as u64, buf);
    for lit in &non_strings {
        encode_literal_into(lit, buf);
    }

    // Encode strings next.
    encode_uleb128_into(strings.len() as u64, buf);

    // Encode string lengths and contents separately.
    for s in &strings {
        encode_uleb128_into(s.len() as u64, buf);
    }
    for s in &strings {
        buf.extend(s.as_bytes());
    }
}

/// Encode a slice of literal into a byte buffer.
pub fn encode_literals(literals: &[Literal]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(literals.len().saturating_mul(8).saturating_add(8));
    encode_literals_into(literals, &mut buf);
    buf
}

/// Write instruction and literal buffers to a file.
pub fn write_instrs(
    instrs: &[OpCode],
    literals: &[Literal],
    file_name: &str,
    compression_level: i32,
) -> std::io::Result<()> {
    // Create the file and buffer for writing.
    let mut file = std::fs::File::create(file_name)?;
    let mut buf = Vec::with_capacity(
        literals
            .len()
            .saturating_mul(8)
            .saturating_add(instrs.len().saturating_mul(4))
            .saturating_add(16),
    );

    // Encoding the literals and instructions into buffers.
    let mut lits_buf = Vec::new();
    encode_literals_into(literals, &mut lits_buf);
    let mut instrs_buf = Vec::new();
    encode_instrs_into(instrs, &mut instrs_buf);

    // Print the uncompressed size of literals and instructions for debugging.
    debug!(
        "[DEBUG_SIZE] Literals uncompressed: {} bytes",
        lits_buf.len()
    );
    debug!(
        "[DEBUG_SIZE] Instructions uncompressed: {} bytes",
        instrs_buf.len()
    );

    // buf.extend_from_slice(&lits_buf);
    // buf.extend_from_slice(&instrs_buf);

    fn compress_buffer(buf: &[u8], compression_level: i32) -> std::io::Result<Vec<u8>> {
        let compressed_result = zstd::encode_all(buf, compression_level)?;
        Ok(compressed_result)
    }

    // Compress the buffer using DEFLATE.
    let comp_lits = compress_buffer(&lits_buf, compression_level)?;
    let comp_instrs = compress_buffer(&instrs_buf, compression_level)?;

    buf.extend_from_slice(&comp_lits);
    buf.extend_from_slice(&comp_instrs);
    let compressed_buf = compress_buffer(&buf, compression_level)?;

    // Print the compressed size of the buffer for debugging.
    debug!(
        "[DEBUG_SIZE] Compressed size: {} bytes",
        compressed_buf.len()
    );

    // Count the occurrences of each opcode for debugging.
    let mut opcode_counts = std::collections::HashMap::new();
    for instr in instrs {
        let name = match instr {
            OpCode::MakeFunction { .. } => "MakeFunction",
            OpCode::MakeField { .. } => "MakeField",
            OpCode::MakeObject { .. } => "MakeObject",
            OpCode::MakeArray { .. } => "MakeArray",
            OpCode::MakeInteger { .. } => "MakeInteger",
            OpCode::MakeIntegerImm { .. } => "MakeIntegerImm",
            OpCode::MakeString { .. } => "MakeString",
            OpCode::MakeNull => "MakeNull",
            OpCode::MakeBoolean { .. } => "MakeBoolean",
            OpCode::CallFunction { .. } => "CallFunction",
            OpCode::StoreGlobal { .. } => "StoreGlobal",
            OpCode::StoreLocal { .. } => "StoreLocal",
            OpCode::Add => "Add",
            OpCode::LoadGlobal { .. } => "LoadGlobal",
            OpCode::LoadLocal { .. } => "LoadLocal",
            OpCode::Nop => "Nop",
            OpCode::MakeFloat { .. } => "MakeFloat",
            OpCode::AddAddr { .. } => "AddAddr",
            OpCode::AddAddr2 { .. } => "AddAddr2",
            OpCode::MakeFieldFromAddr { .. } => "MakeFieldFromAddr",
            OpCode::AddGlobalLeft { .. } => "AddGlobalLeft",
            OpCode::AddGlobalRight { .. } => "AddGlobalRight",
            OpCode::AddGlobal2 { .. } => "AddGlobal2",
            OpCode::MakeFieldFromGlobal { .. } => "MakeFieldFromGlobal",
            OpCode::MakeArray2 { .. } => "MakeArray2",
            OpCode::MakeFieldFromIntegerImm { .. } => "MakeFieldFromIntegerImm",
            OpCode::MakeFieldFromBoolean { .. } => "MakeFieldFromBoolean",
            OpCode::MakePairArray { .. } => "MakePairArray",
            OpCode::MakeFieldIntegerBlock { .. } => "MakeFieldIntegerBlock",
            OpCode::MakeFieldIntegerBlockSequential { .. } => "MakeFieldIntegerBlockSequential",
            OpCode::MakeFieldGlobalBlock { .. } => "MakeFieldGlobalBlock",
            OpCode::MakeFieldAddrBlock { .. } => "MakeFieldAddrBlock",
            OpCode::MakeFieldBooleanBlock { .. } => "MakeFieldBooleanBlock",
            OpCode::MakeStringInline { .. } => "MakeStringInline",
            OpCode::MakeStringNumInline { .. } => "MakeStringNumInline",
            OpCode::MakeStringTsInline { .. } => "MakeStringTsInline",
            OpCode::MakeFieldFromLocal { .. } => "MakeFieldFromLocal",
            OpCode::MakeFieldLocalBlock { .. } => "MakeFieldLocalBlock",
        };
        *opcode_counts.entry(name).or_insert(0) += 1;
    }
    let mut sorted_opcodes: Vec<_> = opcode_counts.into_iter().collect();
    sorted_opcodes.sort_by(|a, b| b.1.cmp(&a.1));

    debug!("=== OpCode Counts ===");
    #[cfg(debug_assertions)]
    for (name, count) in &sorted_opcodes {
        debug!("  {}: {}", name, count);
    }

    file.write_all(&compressed_buf)?;

    Ok(())
}
