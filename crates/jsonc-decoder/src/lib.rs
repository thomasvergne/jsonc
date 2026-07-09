use std::collections::HashMap;

use jsonc_bytecode::{Address, Literal, OpCode};
use jsonc_mlir::MLIR;

#[derive(Debug)]
pub enum DecoderError {
    InvalidOpcode(OpCode),
    InvalidInteger,
    InvalidString,
    InvalidLiteral(String),
}

pub struct Decoder {
    pub bytecode: Vec<jsonc_bytecode::OpCode>,
    pub value_pool: Vec<Literal>,
}

impl Decoder {
    pub fn new(bytecode: Vec<jsonc_bytecode::OpCode>, value_pool: Vec<Literal>) -> Self {
        Self {
            bytecode,
            value_pool,
        }
    }

    // Decode unsigned LEB128; returns (value, bytes_consumed)
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

    // Decode signed LEB128 (SLEB128)
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

    pub fn decode_literals(&mut self, bits: &[u8]) -> Result<usize, DecoderError> {
        let mut idx = 0usize;
        let mut value_pool = Vec::new();
        let (len_u, consumed) = self.decode_uleb128(bits, idx)?;
        let len = len_u as usize;
        idx += consumed;

        let mut quantity = 0usize;

        while quantity < len {
            if idx >= bits.len() {
                return Err(DecoderError::InvalidLiteral(
                    "unexpected end of input".to_string(),
                ));
            }
            let value_type = bits[idx];
            idx += 1; // consume type byte

            match value_type {
                0x00 => {
                    // integer literal: SLEB128 encoded
                    let (integer, consumed) = self.decode_sleb128(bits, idx)?;
                    idx += consumed;
                    value_pool.push(Literal::Integer(integer));
                    quantity += 1;
                }

                0x01 => {
                    // string literal: ULEB128 length + bytes
                    let (str_len_u, consumed) = self.decode_uleb128(bits, idx)?;
                    idx += consumed;
                    let str_len = str_len_u as usize;
                    if idx + str_len > bits.len() {
                        return Err(DecoderError::InvalidString);
                    }
                    let string = self.decode_string(bits, idx, str_len)?;
                    // push the string directly (avoid an extra clone)
                    value_pool.push(Literal::String(string));
                    idx += str_len;
                    quantity += 1;
                }
                0x02 => {
                    // float literal: 8 bytes raw
                    if idx + 8 > bits.len() {
                        return Err(DecoderError::InvalidLiteral("unexpected end of input".to_string()));
                    }
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&bits[idx..idx + 8]);
                    idx += 8;
                    value_pool.push(Literal::Float(u64::from_le_bytes(bytes)));
                    quantity += 1;
                }

                // Not a literal type -> error
                other => {
                    return Err(DecoderError::InvalidLiteral(format!(
                        "Unknown literal type: {}",
                        other
                    )));
                }
            }
        }

        // Save parsed literals into the decoder
        self.value_pool = value_pool;

        Ok(idx)
    }

    pub fn decode(&mut self, bits: Vec<u8>) -> Result<(), DecoderError> {
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

        // Helper that executes a range of instructions and returns the stack after executing
        fn exec_range<'a>(
            decoder: &'a Decoder,
            instrs: &'a [jsonc_bytecode::OpCode],
            start: usize,
            count: usize,
            globals: &mut Vec<MLIR<'a>>,
            global_name_cache: &mut HashMap<usize, &'a str>,
            local_name_cache: &mut HashMap<usize, &'a str>,
            function_name_cache: &mut HashMap<usize, &'a str>,
        ) -> Result<(Vec<MLIR<'a>>, usize), DecoderError> {
            use jsonc_bytecode::OpCode::*;
            use jsonc_mlir::MLIR;

            let mut stack: Vec<MLIR<'a>> = Vec::new();
            let mut functions = HashMap::new();
            let mut i = start;
            let mut processed = 0usize;

            while processed < count {
                if i >= instrs.len() {
                    return Err(DecoderError::InvalidOpcode(OpCode::Nop));
                }
                let instr = instrs[i];
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

                    MakeNull => {
                        stack.push(MLIR::Null);
                    }

                    MakeBoolean { value: idx } => {
                        stack.push(MLIR::Bool(idx));
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
                        let func = match stack.pop() {
                            Some(func) => func,
                            None => {
                                functions.get(&(func_addr as usize)).cloned().ok_or(
                                    DecoderError::InvalidLiteral(format!(
                                        "CallFunction at instr {}: unknown function index {}",
                                        i - 1,
                                        func_addr
                                    )),
                                )?
                            }
                        };

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

        let (stack, _) = exec_range(
            self,
            &self.bytecode,
            0,
            self.bytecode.len(),
            &mut globals,
            &mut global_name_cache,
            &mut local_name_cache,
            &mut function_name_cache,
        )?;

        let mut mlir = Vec::new();
        mlir.extend(globals);
        mlir.extend(stack);

        Ok(mlir)
    }
}
