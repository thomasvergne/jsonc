use std::{collections::HashMap, io::Cursor};

use jsonc_bytecode::{Address, Literal, OpCode};
use jsonc_mlir::MLIR;

const OPCODE_LOAD_LOCAL_INLINE_BASE: u8 = 0x20; // 0x20..=0x2F
const OPCODE_STORE_LOCAL_INLINE_BASE: u8 = 0x30; // 0x30..=0x3F
const OPCODE_LOAD_GLOBAL_INLINE_BASE: u8 = 0x40; // 0x40..=0x4F
const OPCODE_STORE_GLOBAL_INLINE_BASE: u8 = 0x50; // 0x50..=0x5F
const OPCODE_MAKE_TRUE: u8 = 0x60;
const OPCODE_MAKE_FALSE: u8 = 0x61;
const OPCODE_MAKE_INTEGER_IMM: u8 = 0x62;

#[derive(Debug)]
pub enum DecoderError {
    InvalidOpcode(OpCode),
    InvalidInteger,
    InvalidString,
    InvalidLiteral(String),
}

impl std::fmt::Display for DecoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecoderError::InvalidOpcode(op) => write!(f, "Invalid opcode: {:?}", op),
            DecoderError::InvalidInteger => write!(f, "Invalid integer"),
            DecoderError::InvalidString => write!(f, "Invalid string"),
            DecoderError::InvalidLiteral(s) => write!(f, "{}", s),
        }
    }
}

/// Decoder for JSONC bytecode.
/// It contains the bytecode and value pool used for decoding.
pub struct Decoder {
    pub bytecode: Vec<jsonc_bytecode::OpCode>,
    pub value_pool: Vec<Literal>,
}

impl Decoder {
    /// Creates a new decoder with the given bytecode and value pool.
    pub fn new(bytecode: Vec<jsonc_bytecode::OpCode>, value_pool: Vec<Literal>) -> Self {
        Self {
            bytecode,
            value_pool,
        }
    }

    /// Decodes an unsigned LEB128 value from the given bits starting at the given index.
    /// Returns the decoded value and the number of bytes consumed.
    pub fn decode_uleb128(&self, bits: &[u8], idx: usize) -> Result<(u64, usize), DecoderError> {
        let mut result: u64 = 0;
        let mut shift = 0;
        let mut i = idx;
        loop {
            if i >= bits.len() {
                return Err(DecoderError::InvalidInteger);
            }
            let byte = bits[i];
            let low = (byte & 0x7F) as u64;
            result |= low << shift;
            i += 1;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                return Err(DecoderError::InvalidInteger);
            }
        }
        Ok((result, i - idx))
    }

    /// Decodes a signed LEB128 (SLEB128) value from the given bits starting at the given index.
    /// Returns the decoded value and the number of bytes consumed.
    pub fn decode_sleb128(&self, bits: &[u8], idx: usize) -> Result<(i64, usize), DecoderError> {
        let mut result: i64 = 0;
        let mut shift = 0;
        let mut i = idx;
        let mut byte: u8;
        loop {
            if i >= bits.len() {
                return Err(DecoderError::InvalidInteger);
            }
            byte = bits[i];
            let low = (byte & 0x7F) as i64;
            result |= low << shift;
            shift += 7;
            i += 1;
            if byte & 0x80 == 0 {
                break;
            }
            if shift >= 64 {
                return Err(DecoderError::InvalidInteger);
            }
        }
        // sign extend if necessary
        if (shift < 64) && (byte & 0x40) != 0 {
            result |= (!0i64) << shift;
        }
        Ok((result, i - idx))
    }

    /// Decodes a string from the given bits starting at the given index with the given length.
    /// Returns the decoded string.
    pub fn decode_string(
        &self,
        bits: &[u8],
        idx: usize,
        len: usize,
    ) -> Result<String, DecoderError> {
        if idx + len > bits.len() {
            return Err(DecoderError::InvalidString);
        }
        // Prefer from_utf8 to avoid the replacement branch of from_utf8_lossy
        match std::str::from_utf8(&bits[idx..idx + len]) {
            Ok(s) => Ok(s.to_owned()),
            Err(_) => Err(DecoderError::InvalidString),
        }
    }

    /// Decodes literals from the given bits starting at the given index.
    /// Returns the number of bytes consumed.
    pub fn decode_literals(&mut self, bits: &[u8]) -> Result<usize, DecoderError> {
        let mut idx = 0usize;
        let mut value_pool = Vec::new();

        let (non_str_count_u, consumed) = self.decode_uleb128(bits, idx)?;
        idx += consumed;
        let non_str_count = non_str_count_u as usize;

        let mut quantity = 0usize;
        while quantity < non_str_count {
            if idx >= bits.len() {
                return Err(DecoderError::InvalidLiteral(
                    "unexpected end of input".to_string(),
                ));
            }
            let value_type = bits[idx];
            idx += 1;

            match value_type {
                0x00 => {
                    let (integer, consumed) = self.decode_sleb128(bits, idx)?;
                    idx += consumed;
                    value_pool.push(Literal::Integer(integer));
                    quantity += 1;
                }
                0x01 => {
                    let (str_len_u, consumed) = self.decode_uleb128(bits, idx)?;
                    idx += consumed;
                    let str_len = str_len_u as usize;
                    if idx + str_len > bits.len() {
                        return Err(DecoderError::InvalidString);
                    }
                    let string = self.decode_string(bits, idx, str_len)?;
                    value_pool.push(Literal::String(string));
                    idx += str_len;
                    quantity += 1;
                }
                0x02 => {
                    if idx + 8 > bits.len() {
                        return Err(DecoderError::InvalidLiteral(
                            "unexpected end of input".to_string(),
                        ));
                    }
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&bits[idx..idx + 8]);
                    idx += 8;
                    value_pool.push(Literal::Float(u64::from_le_bytes(bytes)));
                    quantity += 1;
                }
                0x03 => {
                    value_pool.push(Literal::Null);
                    quantity += 1;
                }
                0x04 => {
                    if idx >= bits.len() {
                        return Err(DecoderError::InvalidLiteral(
                            "unexpected end of input".to_string(),
                        ));
                    }
                    let val = bits[idx] != 0;
                    idx += 1;
                    value_pool.push(Literal::Bool(val));
                    quantity += 1;
                }
                0x05 => {
                    let (val, consumed) = self.decode_uleb128(bits, idx)?;
                    idx += consumed;
                    value_pool.push(Literal::String(val.to_string()));
                    quantity += 1;
                }
                0x06 => {
                    let (packed, consumed) = self.decode_uleb128(bits, idx)?;
                    idx += consumed;
                    let year = ((packed >> 46) & 0xFF) + 2000;
                    let month = (packed >> 42) & 0x0F;
                    let day = (packed >> 37) & 0x1F;
                    let hour = (packed >> 32) & 0x1F;
                    let minute = (packed >> 26) & 0x3F;
                    let second = (packed >> 20) & 0x3F;
                    let microseconds = packed & 0xFFFFF;
                    let formatted = format!(
                        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}+00:00",
                        year, month, day, hour, minute, second, microseconds
                    );
                    value_pool.push(Literal::String(formatted));
                    quantity += 1;
                }
                0x10..=0x1F => {
                    let str_len = (value_type - 0x10) as usize;
                    if idx + str_len > bits.len() {
                        return Err(DecoderError::InvalidString);
                    }
                    let string = self.decode_string(bits, idx, str_len)?;
                    value_pool.push(Literal::String(string));
                    idx += str_len;
                    quantity += 1;
                }
                other => {
                    println!("{}", idx);
                    return Err(DecoderError::InvalidLiteral(format!(
                        "Unknown non-string literal type: {}",
                        other
                    )));
                }
            }
        }

        let (str_count_u, consumed) = self.decode_uleb128(bits, idx)?;
        idx += consumed;
        let str_count = str_count_u as usize;

        let mut string_lengths = Vec::with_capacity(str_count);
        for _ in 0..str_count {
            let (str_len_u, consumed) = self.decode_uleb128(bits, idx)?;
            idx += consumed;
            string_lengths.push(str_len_u as usize);
        }

        for str_len in string_lengths {
            if idx + str_len > bits.len() {
                return Err(DecoderError::InvalidString);
            }
            let string = self.decode_string(bits, idx, str_len)?;
            idx += str_len;
            value_pool.push(Literal::String(string));
        }

        // Save parsed literals into the decoder
        self.value_pool = value_pool;

        Ok(idx)
    }

    /// Decodes the bytecode from the given bits.
    /// Returns an error if the bytecode is invalid.
    pub fn decode(&mut self, bits: Vec<u8>) -> Result<(), DecoderError> {
        let decompressed_result = zstd::decode_all(Cursor::new(bits));

        let Ok(bits) = decompressed_result else {
            return Err(DecoderError::InvalidLiteral(
                "failed to decompress bytecode".to_string(),
            ));
        };

        let bits = match zstd::decode_all(Cursor::new(&bits)) {
            Ok(decompressed) => decompressed,
            Err(_) => bits,
        };

        let mut idx = 0usize;

        // avoid cloning the whole buffer by passing a slice
        let literals_len = self.decode_literals(&bits)?;
        idx += literals_len;

        let (len_u, consumed) = self.decode_uleb128(&bits, idx)?;
        let len = len_u as usize;
        idx += consumed;
        let mut quantity = 0usize;

        while quantity < len {
            if idx >= bits.len() {
                return Err(DecoderError::InvalidOpcode(OpCode::Nop));
            }
            let opcode = bits[idx];
            idx += 1;

            // Read operands according to opcode using ULEB128 for indexes/lengths
            let op = match opcode {
                0x00 => {
                    // MakeFunction { num_params: u8, body_len: u16 }
                    let (num_params_u, c1) = self.decode_uleb128(&bits, idx)?;
                    idx += c1;
                    let (body_len_u, c2) = self.decode_uleb128(&bits, idx)?;
                    idx += c2;
                    OpCode::MakeFunction {
                        num_params: num_params_u as u8,
                        body_len: body_len_u as u16,
                    }
                }
                0x01 => {
                    let (field_idx, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::MakeField {
                        field_name: field_idx as Address,
                    }
                }
                0x02 => {
                    let (num_fields_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::MakeObject {
                        num_fields: num_fields_u as u32,
                    }
                }
                0x03 => {
                    let (num_elems_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;

                    OpCode::MakeArray {
                        num_elements: num_elems_u as u32,
                    }
                }
                0x04 => {
                    let (val_idx_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::MakeInteger {
                        value: val_idx_u as Address,
                    }
                }
                0x05 => {
                    let (val_idx_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::MakeString {
                        value: val_idx_u as Address,
                    }
                }
                0x06 => OpCode::MakeNull,
                0x07 => {
                    let (b_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::MakeBoolean { value: b_u != 0 }
                }
                0x08 => {
                    let (num_args_u, c1) = self.decode_uleb128(&bits, idx)?;
                    idx += c1;
                    let (func_idx_u, c2) = self.decode_uleb128(&bits, idx)?;
                    idx += c2;
                    OpCode::CallFunction {
                        num_args: num_args_u as u8,
                        func_index: func_idx_u as Address,
                    }
                }
                0x09 => {
                    let (var_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::StoreGlobal {
                        var_index: var_u as Address,
                    }
                }
                0x0A => {
                    let (var_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::StoreLocal {
                        var_index: var_u as Address,
                    }
                }
                0x0B => OpCode::Add,
                0x0C => {
                    let (var_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::LoadGlobal {
                        var_index: var_u as Address,
                    }
                }
                0x0D => {
                    let (var_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::LoadLocal {
                        var_index: var_u as Address,
                    }
                }
                0x0E => OpCode::Nop,
                0x0F => {
                    let (val_idx_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::MakeFloat {
                        value: val_idx_u as Address,
                    }
                }
                0x10 => {
                    let (addr_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::AddAddr {
                        addr: addr_u as Address,
                    }
                }
                0x11 => {
                    let (addr1_u, c1) = self.decode_uleb128(&bits, idx)?;
                    idx += c1;
                    let (addr2_u, c2) = self.decode_uleb128(&bits, idx)?;
                    idx += c2;
                    OpCode::AddAddr2 {
                        addr1: addr1_u as Address,
                        addr2: addr2_u as Address,
                    }
                }
                0x12 => {
                    let (field_name_u, c1) = self.decode_uleb128(&bits, idx)?;
                    idx += c1;
                    let (addr_u, c2) = self.decode_uleb128(&bits, idx)?;
                    idx += c2;
                    OpCode::MakeFieldFromAddr {
                        field_name: field_name_u as Address,
                        addr: addr_u as Address,
                    }
                }
                0x13 => {
                    let (addr_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::AddGlobalLeft {
                        addr: addr_u as Address,
                    }
                }
                0x1F => {
                    let (addr_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::AddGlobalRight {
                        addr: addr_u as Address,
                    }
                }
                0x14 => {
                    let (addr1_u, c1) = self.decode_uleb128(&bits, idx)?;
                    idx += c1;
                    let (addr2_u, c2) = self.decode_uleb128(&bits, idx)?;
                    idx += c2;
                    OpCode::AddGlobal2 {
                        addr1: addr1_u as Address,
                        addr2: addr2_u as Address,
                    }
                }
                0x15 => {
                    let (field_name_u, c1) = self.decode_uleb128(&bits, idx)?;
                    idx += c1;
                    let (addr_u, c2) = self.decode_uleb128(&bits, idx)?;
                    idx += c2;
                    OpCode::MakeFieldFromGlobal {
                        field_name: field_name_u as Address,
                        addr: addr_u as Address,
                    }
                }
                0x16 => {
                    let (addr1_u, c1) = self.decode_uleb128(&bits, idx)?;
                    idx += c1;
                    let (addr2_u, c2) = self.decode_uleb128(&bits, idx)?;
                    idx += c2;
                    OpCode::MakeArray2 {
                        addr1: addr1_u as Address,
                        addr2: addr2_u as Address,
                    }
                }
                0x17 => {
                    let (field_name_u, c1) = self.decode_uleb128(&bits, idx)?;
                    idx += c1;
                    let (value, c2) = self.decode_sleb128(&bits, idx)?;
                    idx += c2;
                    OpCode::MakeFieldFromIntegerImm {
                        field_name: field_name_u as Address,
                        value,
                    }
                }
                0x18 => {
                    let (field_name_u, c1) = self.decode_uleb128(&bits, idx)?;
                    idx += c1;
                    if idx >= bits.len() {
                        return Err(DecoderError::InvalidInteger);
                    }
                    let val = bits[idx];
                    idx += 1;
                    OpCode::MakeFieldFromBoolean {
                        field_name: field_name_u as Address,
                        value: val != 0,
                    }
                }
                0x19 => {
                    let (len, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    let mut pairs = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        let (addr1_u, c1) = self.decode_uleb128(&bits, idx)?;
                        idx += c1;
                        let (addr2_u, c2) = self.decode_uleb128(&bits, idx)?;
                        idx += c2;
                        pairs.push((addr1_u as Address, addr2_u as Address));
                    }
                    OpCode::MakePairArray { pairs }
                }
                0x1A => {
                    let (len, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    let mut fields = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        let (field_name_u, c1) = self.decode_uleb128(&bits, idx)?;
                        idx += c1;
                        let (value, c2) = self.decode_sleb128(&bits, idx)?;
                        idx += c2;
                        fields.push((field_name_u as Address, value));
                    }
                    OpCode::MakeFieldIntegerBlock { fields }
                }
                0x1B => {
                    let (count, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    let (start_value, c2) = self.decode_sleb128(&bits, idx)?;
                    idx += c2;
                    let (start_field_address, c3) = self.decode_uleb128(&bits, idx)?;
                    idx += c3;
                    OpCode::MakeFieldIntegerBlockSequential {
                        count: count as u32,
                        start_value,
                        start_field_address: start_field_address as Address,
                    }
                }
                0x1C => {
                    let (len, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    let mut fields = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        let (field_name_u, c1) = self.decode_uleb128(&bits, idx)?;
                        idx += c1;
                        let (addr_u, c2) = self.decode_uleb128(&bits, idx)?;
                        idx += c2;
                        fields.push((field_name_u as Address, addr_u as Address));
                    }
                    OpCode::MakeFieldGlobalBlock { fields }
                }
                0x1D => {
                    let (len, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    let mut fields = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        let (field_name_u, c1) = self.decode_uleb128(&bits, idx)?;
                        idx += c1;
                        let (addr_u, c2) = self.decode_uleb128(&bits, idx)?;
                        idx += c2;
                        fields.push((field_name_u as Address, addr_u as Address));
                    }
                    OpCode::MakeFieldAddrBlock { fields }
                }
                0x1E => {
                    let (len, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    let mut fields = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        let (field_name_u, c1) = self.decode_uleb128(&bits, idx)?;
                        idx += c1;
                        if idx >= bits.len() {
                            return Err(DecoderError::InvalidInteger);
                        }
                        let val = bits[idx];
                        idx += 1;
                        fields.push((field_name_u as Address, val != 0));
                    }
                    OpCode::MakeFieldBooleanBlock { fields }
                }
                OPCODE_MAKE_INTEGER_IMM => {
                    let (value, c) = self.decode_sleb128(&bits, idx)?;
                    idx += c;
                    OpCode::MakeIntegerImm { value }
                }
                OPCODE_MAKE_TRUE => OpCode::MakeBoolean { value: true },
                OPCODE_MAKE_FALSE => OpCode::MakeBoolean { value: false },
                0x63 => {
                    let (len_u, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    let len = len_u as usize;
                    if idx + len > bits.len() {
                        return Err(DecoderError::InvalidString);
                    }
                    let value = self.decode_string(&bits, idx, len)?;
                    idx += len;
                    OpCode::MakeStringInline { value }
                }
                0x64 => {
                    let (val, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::MakeStringNumInline { value: val }
                }
                0x65 => {
                    let (val, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    OpCode::MakeStringTsInline { value: val }
                }
                0x66 => {
                    let (len, c) = self.decode_uleb128(&bits, idx)?;
                    idx += c;
                    let mut fields = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        let (field_name_u, c1) = self.decode_uleb128(&bits, idx)?;
                        idx += c1;
                        let (addr_u, c2) = self.decode_uleb128(&bits, idx)?;
                        idx += c2;
                        fields.push((field_name_u as Address, addr_u as Address));
                    }
                    OpCode::MakeFieldLocalBlock { fields }
                }
                0x67 => {
                    let (field_name_u, c1) = self.decode_uleb128(&bits, idx)?;
                    idx += c1;
                    let (addr_u, c2) = self.decode_uleb128(&bits, idx)?;
                    idx += c2;
                    OpCode::MakeFieldFromLocal {
                        field_name: field_name_u as Address,
                        addr: addr_u as Address,
                    }
                }

                op @ OPCODE_LOAD_LOCAL_INLINE_BASE..=0x2F => OpCode::LoadLocal {
                    var_index: (op - OPCODE_LOAD_LOCAL_INLINE_BASE) as Address,
                },
                op @ OPCODE_STORE_LOCAL_INLINE_BASE..=0x3F => OpCode::StoreLocal {
                    var_index: (op - OPCODE_STORE_LOCAL_INLINE_BASE) as Address,
                },
                op @ OPCODE_LOAD_GLOBAL_INLINE_BASE..=0x4F => OpCode::LoadGlobal {
                    var_index: (op - OPCODE_LOAD_GLOBAL_INLINE_BASE) as Address,
                },
                op @ OPCODE_STORE_GLOBAL_INLINE_BASE..=0x5F => OpCode::StoreGlobal {
                    var_index: (op - OPCODE_STORE_GLOBAL_INLINE_BASE) as Address,
                },
                op @ 0x70..=0xFF => OpCode::LoadGlobal {
                    var_index: (op - 0x70) as Address,
                },
                _ => return Err(DecoderError::InvalidOpcode(OpCode::Nop)),
            };

            self.bytecode.push(op);
            quantity += 1;
        }

        Ok(())
    }

    /// Execute the decoded bytecode and produce MLIR nodes.
    ///
    /// The returned MLIR nodes borrow from the decoder's own value pool where needed.
    pub fn to_mlir<'a>(&'a self) -> Result<Vec<MLIR<'a>>, DecoderError> {
        fn cached_symbol<'a>(
            cache: &mut HashMap<usize, &'a str>,
            prefix: &str,
            index: usize,
        ) -> &'a str {
            if let Some(name) = cache.get(&index) {
                return name;
            }
            let name: &'a str = Box::leak(format!("{}{}", prefix, index).into_boxed_str());
            cache.insert(index, name);
            name
        }

        fn literal_to_mlir<'a>(value: Literal) -> MLIR<'a> {
            match value {
                Literal::String(s) => MLIR::String(Box::leak(s.into_boxed_str())),
                Literal::Integer(n) => MLIR::Number(n as f64),
                Literal::Bool(b) => MLIR::Bool(b),
                Literal::Null => MLIR::Null,
                Literal::Float(n) => MLIR::Number(f64::from_bits(n)),
                // _ => MLIR::Null,
            }
        }

        /// Helper that executes a range of instructions and returns the stack after executing
        ///
        /// # Arguments
        ///
        /// * `decoder` - The decoder instance.
        /// * `instrs` - The instructions to execute.
        /// * `start` - The starting index of the range.
        /// * `count` - The number of instructions to execute.
        /// * `globals` - The global variables.
        /// * `functions` - The functions.
        /// * `global_name_cache` - The global name cache.
        /// * `local_name_cache` - The local name cache.
        /// * `function_name_cache` - The function name cache.
        ///
        /// # Returns
        ///
        /// * `Ok((stack, consumed))` - The stack after executing the range and the number of bytes consumed.
        /// * `Err(error)` - An error if the bytecode is invalid.
        fn exec_range<'a>(
            decoder: &'a Decoder,
            instrs: &'a [jsonc_bytecode::OpCode],
            start: usize,
            count: usize,
            globals: &mut Vec<MLIR<'a>>,
            functions: &mut HashMap<usize, MLIR<'a>>,
            global_name_cache: &mut HashMap<usize, &'a str>,
            local_name_cache: &mut HashMap<usize, &'a str>,
            function_name_cache: &mut HashMap<usize, &'a str>,
        ) -> Result<(Vec<MLIR<'a>>, usize), DecoderError> {
            use jsonc_bytecode::OpCode::*;
            use jsonc_mlir::MLIR;

            let mut stack: Vec<MLIR<'a>> = Vec::new();
            let mut i = start;
            let mut processed = 0usize;

            while processed < count {
                if i >= instrs.len() {
                    return Err(DecoderError::InvalidOpcode(OpCode::Nop));
                }
                let instr = instrs[i].clone();
                i += 1;
                processed += 1;

                match instr {
                    MakeInteger { value: idx } => match decoder.value_pool.get(idx as usize) {
                        Some(jsonc_bytecode::Literal::Integer(v)) => {
                            stack.push(MLIR::Number(*v as f64));
                        }
                        _ => {
                            return Err(DecoderError::InvalidLiteral(
                                "MakeInteger: literal is not an integer".to_string(),
                            ));
                        }
                    },
                    MakeIntegerImm { value } => {
                        stack.push(MLIR::Number(value as f64));
                    }

                    MakeFloat { value: idx } => match decoder.value_pool.get(idx as usize) {
                        Some(jsonc_bytecode::Literal::Float(v)) => {
                            stack.push(MLIR::Number(f64::from_bits(*v)));
                        }
                        _ => {
                            return Err(DecoderError::InvalidLiteral(
                                "MakeFloat: literal is not a float".to_string(),
                            ));
                        }
                    },

                    MakeString { value: idx } => match decoder.value_pool.get(idx as usize) {
                        Some(jsonc_bytecode::Literal::String(s)) => {
                            stack.push(MLIR::String(s.as_str()));
                        }
                        _ => {
                            return Err(DecoderError::InvalidLiteral(
                                "MakeString: literal is not a string".to_string(),
                            ));
                        }
                    },

                    MakeStringInline { value } => {
                        stack.push(MLIR::String(Box::leak(value.into_boxed_str())));
                    }

                    MakeStringNumInline { value } => {
                        let s = value.to_string();
                        stack.push(MLIR::String(Box::leak(s.into_boxed_str())));
                    }

                    MakeStringTsInline { value } => {
                        let year = ((value >> 46) & 0xFF) + 2000;
                        let month = (value >> 42) & 0x0F;
                        let day = (value >> 37) & 0x1F;
                        let hour = (value >> 32) & 0x1F;
                        let minute = (value >> 26) & 0x3F;
                        let second = (value >> 20) & 0x3F;
                        let microseconds = value & 0xFFFFF;
                        let formatted = format!(
                            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}+00:00",
                            year, month, day, hour, minute, second, microseconds
                        );
                        stack.push(MLIR::String(Box::leak(formatted.into_boxed_str())));
                    }

                    MakeNull => {
                        stack.push(MLIR::Null);
                    }

                    MakeBoolean { value: idx } => {
                        stack.push(MLIR::Bool(idx));
                    }

                    AddAddr { addr } => {
                        let rhs = stack.pop().unwrap();
                        let lhs = decoder.value_pool[addr as usize].clone();
                        stack.push(MLIR::Add {
                            left: Box::new(literal_to_mlir(lhs)),
                            right: Box::new(rhs),
                        });
                    }

                    AddAddr2 { addr1, addr2 } => {
                        let rhs = decoder.value_pool[addr2 as usize].clone();
                        let lhs = decoder.value_pool[addr1 as usize].clone();
                        stack.push(MLIR::Add {
                            left: Box::new(literal_to_mlir(lhs)),
                            right: Box::new(literal_to_mlir(rhs)),
                        });
                    }
                    AddGlobalLeft { addr } => {
                        let rhs = stack.pop().ok_or(DecoderError::InvalidLiteral(format!(
                            "AddGlobalLeft: stack is too small"
                        )))?;
                        let name = cached_symbol(global_name_cache, "g", addr as usize);
                        stack.push(MLIR::Add {
                            left: Box::new(MLIR::Variable(name)),
                            right: Box::new(rhs),
                        });
                    }

                    AddGlobalRight { addr } => {
                        let lhs = stack.pop().ok_or(DecoderError::InvalidLiteral(format!(
                            "AddGlobalRight: stack is too small"
                        )))?;
                        let name = cached_symbol(global_name_cache, "g", addr as usize);
                        stack.push(MLIR::Add {
                            left: Box::new(lhs),
                            right: Box::new(MLIR::Variable(name)),
                        });
                    }

                    MakeFieldFromGlobal { field_name, addr } => {
                        let name = cached_symbol(global_name_cache, "g", addr as usize);
                        let field_name = decoder.value_pool[field_name as usize].clone();

                        if let Literal::String(str) = &field_name {
                            stack.push(MLIR::Object(vec![(
                                Box::leak(str.clone().into_boxed_str()),
                                MLIR::Variable(name),
                            )]));
                        } else {
                            return Err(DecoderError::InvalidString);
                        }
                    }

                    MakeFieldFromLocal { field_name, addr } => {
                        let name = cached_symbol(local_name_cache, "l", addr as usize);
                        let field_name = decoder.value_pool[field_name as usize].clone();

                        if let Literal::String(str) = &field_name {
                            stack.push(MLIR::Object(vec![(
                                Box::leak(str.clone().into_boxed_str()),
                                MLIR::Variable(name),
                            )]));
                        } else {
                            return Err(DecoderError::InvalidString);
                        }
                    }

                    AddGlobal2 { addr1, addr2 } => {
                        let name1 = cached_symbol(global_name_cache, "g", addr1 as usize);
                        let name2 = cached_symbol(global_name_cache, "g", addr2 as usize);

                        stack.push(MLIR::Add {
                            left: Box::new(MLIR::Variable(name1)),
                            right: Box::new(MLIR::Variable(name2)),
                        });
                    }

                    MakeFieldFromAddr { field_name, addr } => {
                        let field_name = decoder.value_pool[field_name as usize].clone();
                        let value = decoder.value_pool[addr as usize].clone();

                        let field_name = literal_to_mlir(field_name);
                        let value = literal_to_mlir(value);

                        match field_name {
                            MLIR::String(str) => {
                                stack.push(MLIR::Object(vec![(str, value)]));
                            }

                            _ => {
                                return Err(DecoderError::InvalidLiteral(format!(
                                    "MakeFieldFromAddr at instr {}: field_name is not a string",
                                    i - 1
                                )));
                            }
                        }
                    }

                    MakeFieldFromIntegerImm { field_name, value } => {
                        let field_name = decoder.value_pool[field_name as usize].clone();
                        if let Literal::String(str) = &field_name {
                            stack.push(MLIR::Object(vec![(
                                Box::leak(str.clone().into_boxed_str()),
                                MLIR::Number(value as f64),
                            )]));
                        } else {
                            return Err(DecoderError::InvalidString);
                        }
                    }

                    MakeFieldFromBoolean { field_name, value } => {
                        let field_name = decoder.value_pool[field_name as usize].clone();
                        if let Literal::String(str) = &field_name {
                            stack.push(MLIR::Object(vec![(
                                Box::leak(str.clone().into_boxed_str()),
                                MLIR::Bool(value),
                            )]));
                        } else {
                            return Err(DecoderError::InvalidString);
                        }
                    }

                    MakePairArray { pairs } => {
                        let mut elements = Vec::with_capacity(pairs.len());
                        for (addr1, addr2) in pairs.iter().copied() {
                            let val1 = decoder.value_pool.get(addr1 as usize).cloned().ok_or_else(
                                || {
                                    DecoderError::InvalidLiteral(
                                        "MakePairArray: addr1 out of bounds".to_string(),
                                    )
                                },
                            )?;
                            let val2 = decoder.value_pool.get(addr2 as usize).cloned().ok_or_else(
                                || {
                                    DecoderError::InvalidLiteral(
                                        "MakePairArray: addr2 out of bounds".to_string(),
                                    )
                                },
                            )?;
                            elements.push(MLIR::Array(vec![
                                literal_to_mlir(val1),
                                literal_to_mlir(val2),
                            ]));
                        }
                        stack.push(MLIR::Array(elements));
                    }

                    MakeFieldIntegerBlock { fields } => {
                        for (field_name, value) in fields.iter().copied() {
                            let field_name_lit = decoder
                                .value_pool
                                .get(field_name as usize)
                                .cloned()
                                .ok_or_else(|| {
                                    DecoderError::InvalidLiteral(
                                        "MakeFieldIntegerBlock: field_name out of bounds"
                                            .to_string(),
                                    )
                                })?;
                            if let Literal::String(str) = &field_name_lit {
                                stack.push(MLIR::Object(vec![(
                                    Box::leak(str.clone().into_boxed_str()),
                                    MLIR::Number(value as f64),
                                )]));
                            } else {
                                return Err(DecoderError::InvalidString);
                            }
                        }
                    }

                    MakeFieldIntegerBlockSequential {
                        count,
                        start_value,
                        start_field_address,
                    } => {
                        let mut current_value = start_value;
                        for i in 0..count {
                            let field_name = start_field_address + i;
                            let field_name_lit = decoder
                                .value_pool
                                .get(field_name as usize)
                                .cloned()
                                .ok_or_else(|| {
                                    DecoderError::InvalidLiteral(
                                        "MakeFieldIntegerBlockSequential: field_name out of bounds"
                                            .to_string(),
                                    )
                                })?;
                            if let Literal::String(str) = &field_name_lit {
                                stack.push(MLIR::Object(vec![(
                                    Box::leak(str.clone().into_boxed_str()),
                                    MLIR::Number(current_value as f64),
                                )]));
                                current_value += 1;
                            } else {
                                return Err(DecoderError::InvalidString);
                            }
                        }
                    }

                    MakeFieldGlobalBlock { fields } => {
                        for (field_name, addr) in fields.iter().copied() {
                            let name = cached_symbol(global_name_cache, "g", addr as usize);
                            let field_name_lit = decoder
                                .value_pool
                                .get(field_name as usize)
                                .cloned()
                                .ok_or_else(|| {
                                    DecoderError::InvalidLiteral(
                                        "MakeFieldGlobalBlock: field_name out of bounds"
                                            .to_string(),
                                    )
                                })?;
                            if let Literal::String(str) = &field_name_lit {
                                stack.push(MLIR::Object(vec![(
                                    Box::leak(str.clone().into_boxed_str()),
                                    MLIR::Variable(name),
                                )]));
                            } else {
                                return Err(DecoderError::InvalidString);
                            }
                        }
                    }

                    MakeFieldLocalBlock { fields } => {
                        for (field_name, addr) in fields.iter().copied() {
                            let name = cached_symbol(local_name_cache, "l", addr as usize);
                            let field_name_lit = decoder
                                .value_pool
                                .get(field_name as usize)
                                .cloned()
                                .ok_or_else(|| {
                                    DecoderError::InvalidLiteral(
                                        "MakeFieldLocalBlock: field_name out of bounds".to_string(),
                                    )
                                })?;
                            if let Literal::String(str) = &field_name_lit {
                                stack.push(MLIR::Object(vec![(
                                    Box::leak(str.clone().into_boxed_str()),
                                    MLIR::Variable(name),
                                )]));
                            } else {
                                return Err(DecoderError::InvalidString);
                            }
                        }
                    }

                    MakeFieldAddrBlock { fields } => {
                        for (field_name, addr) in fields.iter().copied() {
                            let field_name_lit = decoder
                                .value_pool
                                .get(field_name as usize)
                                .cloned()
                                .ok_or_else(|| {
                                    DecoderError::InvalidLiteral(
                                        "MakeFieldAddrBlock: field_name out of bounds".to_string(),
                                    )
                                })?;
                            let value_lit =
                                decoder.value_pool.get(addr as usize).cloned().ok_or_else(
                                    || {
                                        DecoderError::InvalidLiteral(
                                            "MakeFieldAddrBlock: addr out of bounds".to_string(),
                                        )
                                    },
                                )?;
                            if let Literal::String(str) = &field_name_lit {
                                stack.push(MLIR::Object(vec![(
                                    Box::leak(str.clone().into_boxed_str()),
                                    literal_to_mlir(value_lit),
                                )]));
                            } else {
                                return Err(DecoderError::InvalidString);
                            }
                        }
                    }

                    MakeFieldBooleanBlock { fields } => {
                        for (field_name, value) in fields.iter().copied() {
                            let field_name_lit = decoder
                                .value_pool
                                .get(field_name as usize)
                                .cloned()
                                .ok_or_else(|| {
                                    DecoderError::InvalidLiteral(
                                        "MakeFieldBooleanBlock: field_name out of bounds"
                                            .to_string(),
                                    )
                                })?;
                            if let Literal::String(str) = &field_name_lit {
                                stack.push(MLIR::Object(vec![(
                                    Box::leak(str.clone().into_boxed_str()),
                                    MLIR::Bool(value),
                                )]));
                            } else {
                                return Err(DecoderError::InvalidString);
                            }
                        }
                    }

                    MakeArray { num_elements: len } => {
                        if stack.len() < len as usize {
                            return Err(DecoderError::InvalidLiteral(format!(
                                "MakeArray at instr {}: stack is too small (need {} have {})",
                                i - 1,
                                len,
                                stack.len()
                            )));
                        }

                        let mut elements = Vec::with_capacity(len as usize);
                        for _ in 0..len {
                            elements.push(stack.pop().unwrap());
                        }
                        elements.reverse();
                        stack.push(MLIR::Array(elements));
                    }

                    MakeArray2 { addr1, addr2 } => {
                        let val1 =
                            decoder
                                .value_pool
                                .get(addr1 as usize)
                                .cloned()
                                .ok_or_else(|| {
                                    DecoderError::InvalidLiteral(
                                        "MakeArray2: addr1 out of bounds".to_string(),
                                    )
                                })?;
                        let val2 =
                            decoder
                                .value_pool
                                .get(addr2 as usize)
                                .cloned()
                                .ok_or_else(|| {
                                    DecoderError::InvalidLiteral(
                                        "MakeArray2: addr2 out of bounds".to_string(),
                                    )
                                })?;
                        stack.push(MLIR::Array(vec![
                            literal_to_mlir(val1),
                            literal_to_mlir(val2),
                        ]));
                    }

                    MakeField {
                        field_name: key_idx,
                    } => {
                        if stack.is_empty() {
                            return Err(DecoderError::InvalidLiteral(format!(
                                "MakeField at instr {}: stack is empty",
                                i - 1
                            )));
                        }
                        let value = stack.pop().unwrap();
                        match decoder.value_pool.get(key_idx as usize) {
                            Some(jsonc_bytecode::Literal::String(k)) => {
                                stack.push(MLIR::Object(vec![(k.as_str(), value)]));
                            }
                            _ => {
                                return Err(DecoderError::InvalidLiteral(
                                    "MakeField: literal is not a string".to_string(),
                                ));
                            }
                        }
                    }

                    MakeObject { num_fields: len } => {
                        let mut pairs: Vec<(&'a str, MLIR<'a>)> = Vec::with_capacity(len as usize);
                        for _ in 0..len {
                            let top = stack.pop().ok_or(DecoderError::InvalidLiteral(format!(
                                "MakeObject at instr {}: stack is too small",
                                i - 1
                            )))?;
                            match top {
                                MLIR::Object(obj) => {
                                    pairs.extend(obj);
                                }
                                _ => {
                                    return Err(DecoderError::InvalidLiteral(format!(
                                        "MakeObject at instr {}: expected single-field object on stack",
                                        i - 1
                                    )));
                                }
                            }
                        }
                        pairs.reverse();
                        stack.push(MLIR::Object(pairs));
                    }

                    Add => {
                        let right = stack.pop().ok_or(DecoderError::InvalidLiteral(format!(
                            "Add at instr {}: stack is too small",
                            i - 1
                        )))?;
                        let left = stack.pop().ok_or(DecoderError::InvalidLiteral(format!(
                            "Add at instr {}: stack is too small",
                            i - 1
                        )))?;
                        stack.push(MLIR::Add {
                            left: Box::new(left),
                            right: Box::new(right),
                        });
                    }

                    StoreGlobal { var_index: idx } => {
                        let idx = idx as usize;
                        let value = stack.pop().ok_or(DecoderError::InvalidLiteral(format!(
                            "StoreGlobal at instr {}: stack is too small",
                            i - 1
                        )))?;
                        // ensure globals vector can hold idx
                        if globals.len() <= idx {
                            globals.resize(idx + 1, MLIR::Null);
                        }
                        globals[idx] = value.clone();

                        // create a let binding name v{idx}
                        let name = cached_symbol(global_name_cache, "g", idx);
                        let let_node = MLIR::Let {
                            name,
                            value: Box::new(value),
                        };
                        stack.push(let_node);
                    }

                    StoreLocal { var_index: idx } => {
                        // For decoding we treat local stores like globals but do not persist across ranges
                        let idx = idx as usize;
                        let value = stack.pop().ok_or(DecoderError::InvalidLiteral(format!(
                            "StoreLocal at instr {}: stack is too small",
                            i - 1
                        )))?;
                        let name = cached_symbol(local_name_cache, "l", idx);
                        let let_node = MLIR::Let {
                            name,
                            value: Box::new(value),
                        };
                        stack.push(let_node);
                    }

                    LoadGlobal { var_index: idx } => {
                        let idx = idx as usize;
                        let name = cached_symbol(global_name_cache, "g", idx);
                        stack.push(MLIR::Variable(name));
                    }

                    LoadLocal { var_index: idx } => {
                        let idx = idx as usize;
                        let name = cached_symbol(local_name_cache, "l", idx);
                        stack.push(MLIR::Variable(name));
                    }

                    MakeFunction {
                        num_params: params_len,
                        body_len,
                    } => {
                        let func_addr = i;
                        // execute body instructions in a nested fresh execution to obtain body MLIR nodes
                        let (body_stack, consumed) = exec_range(
                            decoder,
                            instrs,
                            i,
                            body_len as usize,
                            globals,
                            functions,
                            global_name_cache,
                            local_name_cache,
                            function_name_cache,
                        )?;
                        // advance i and processed by body_len
                        i += consumed;
                        processed += consumed;

                        // create placeholder param names
                        let mut params: Vec<&'a str> = Vec::new();
                        for p in 0..params_len as usize {
                            let name = cached_symbol(local_name_cache, "l", p);
                            params.push(name);
                        }

                        let func_name = cached_symbol(function_name_cache, "f", func_addr);

                        let body = match body_stack.as_slice() {
                            [] => Box::new(MLIR::Null),
                            [x] => Box::new(x.clone()),
                            _ => Box::new(MLIR::Array(body_stack)),
                        };
                        globals.push(MLIR::MakeFunction {
                            name: func_name,
                            params: params.clone(),
                            body: body.clone(),
                        });

                        functions.insert(
                            func_addr,
                            MLIR::MakeFunction {
                                name: func_name,
                                params,
                                body: body,
                            },
                        );
                    }

                    CallFunction {
                        num_args: args_len,
                        func_index: func_addr,
                    } => {
                        if stack.len() < args_len as usize {
                            return Err(DecoderError::InvalidLiteral(format!(
                                "CallFunction at instr {}: stack is too small",
                                i - 1
                            )));
                        }
                        let args = stack.split_off(stack.len() - args_len as usize);
                        let func = functions.get(&(func_addr as usize)).cloned().ok_or(
                            DecoderError::InvalidLiteral(format!(
                                "CallFunction at instr {}: unknown function index {}",
                                i - 1,
                                func_addr
                            )),
                        )?;

                        let func_name = match func {
                            MLIR::MakeFunction { name, .. } => name,
                            _ => return Err(DecoderError::InvalidOpcode(instr)),
                        };
                        stack.push(MLIR::FunctionCall {
                            name: func_name,
                            args,
                        });
                    }

                    _ => return Err(DecoderError::InvalidOpcode(OpCode::Nop)),
                }
            }

            Ok((stack, processed))
        }

        // Execute the whole bytecode
        let mut globals: Vec<MLIR<'a>> = Vec::with_capacity(u8::MAX as usize);
        let mut global_name_cache: HashMap<usize, &'a str> = HashMap::new();
        let mut local_name_cache: HashMap<usize, &'a str> = HashMap::new();
        let mut function_name_cache: HashMap<usize, &'a str> = HashMap::new();

        let mut functions = HashMap::new();
        let (stack, _) = exec_range(
            self,
            &self.bytecode,
            0,
            self.bytecode.len(),
            &mut globals,
            &mut functions,
            &mut global_name_cache,
            &mut local_name_cache,
            &mut function_name_cache,
        )?;

        let mut sorted_functions: Vec<(usize, MLIR<'a>)> = functions.into_iter().collect();
        sorted_functions.sort_by_key(|(addr, _)| *addr);

        let mut mlir = Vec::new();
        for (_, func) in sorted_functions {
            mlir.push(func);
        }
        mlir.extend(stack);

        Ok(mlir)
    }
}
