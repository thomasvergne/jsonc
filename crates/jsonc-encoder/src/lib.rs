use std::io::Write;

use jsonc_bytecode::{Literal, OpCode};

// Encode unsigned LEB128
fn encode_uleb128(mut mut_val: u64) -> Vec<u8> {
    let mut buf = Vec::new();
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
    buf
}

// Encode signed LEB128 (SLEB128)
fn encode_sleb128(mut mut_val: i64) -> Vec<u8> {
    let mut buf = Vec::new();
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
    buf
}

pub fn encode(instr: &OpCode) -> Vec<u8> {
    let mut buf = Vec::new();

    buf.push(instr.opcode());

    match instr {
        OpCode::MakeFunction {
            num_params,
            body_len,
        } => {
            buf.extend(encode_uleb128(*num_params as u64));
            buf.extend(encode_uleb128(*body_len as u64));
        }
        OpCode::MakeField { field_name } => {
            buf.extend(encode_uleb128(*field_name as u64));
        }
        OpCode::MakeObject { num_fields } => {
            buf.extend(encode_uleb128(*num_fields as u64));
        }
        OpCode::MakeArray { num_elements } => {
            buf.extend(encode_uleb128(*num_elements as u64));
        }
        OpCode::MakeInteger { value } => {
            buf.extend(encode_uleb128(*value as u64));
        }
        OpCode::MakeString { value } => {
            buf.extend(encode_uleb128(*value as u64));
        }
        OpCode::MakeNull => {}
        OpCode::MakeBoolean { value } => {
            buf.extend(encode_uleb128(if *value { 1 } else { 0 }));
        }
        OpCode::CallFunction {
            num_args,
            func_index,
        } => {
            buf.extend(encode_uleb128(*num_args as u64));
            buf.extend(encode_uleb128(*func_index as u64));
        }
        OpCode::StoreGlobal { var_index } => {
            buf.extend(encode_uleb128(*var_index as u64));
        }
        OpCode::StoreLocal { var_index } => {
            buf.extend(encode_uleb128(*var_index as u64));
        }
        OpCode::Add => {}
        OpCode::LoadGlobal { var_index } => {
            buf.extend(encode_uleb128(*var_index as u64));
        }
        OpCode::LoadLocal { var_index } => {
            buf.extend(encode_uleb128(*var_index as u64));
        }
        OpCode::Nop => {}
    }

    buf
}

pub fn encode_instrs(instrs: &[OpCode]) -> Vec<u8> {
    let mut buf = Vec::new();

    buf.extend(encode_uleb128(instrs.len() as u64));

    for instr in instrs {
        buf.extend(encode(instr));
    }
    buf
}

pub fn encode_literal(literal: &Literal) -> Vec<u8> {
    let mut buf = Vec::new();
    match literal {
        Literal::Integer(i) => {
            buf.push(0x00);
            buf.extend(encode_sleb128(*i));
        }
        Literal::String(s) => {
            buf.push(0x01);
            buf.extend(encode_uleb128(s.len() as u64));
            buf.extend(s.as_bytes());
        }
    }
    buf
}

pub fn encode_literals(literals: &[Literal]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend(encode_uleb128(literals.len() as u64));

    for literal in literals {
        buf.extend(encode_literal(literal));
    }

    buf
}

pub fn write_instrs(
    instrs: &[OpCode],
    literals: &[Literal],
    file_name: &str,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(file_name)?;
    file.write_all(&encode_literals(literals))?;
    file.write_all(&encode_instrs(instrs))?;

    Ok(())
}
