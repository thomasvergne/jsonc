use std::io::Write;

use jsonc_bytecode::{Literal, OpCode};

// Encode unsigned LEB128
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

// Encode signed LEB128 (SLEB128)
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

fn encode_into(instr: &OpCode, buf: &mut Vec<u8>) {
    buf.push(instr.opcode());

    match instr {
        OpCode::MakeFunction {
            num_params,
            body_len,
        } => {
            encode_uleb128_into(*num_params as u64, buf);
            encode_uleb128_into(*body_len as u64, buf);
        }
        OpCode::MakeField { field_name } => {
            encode_uleb128_into(*field_name as u64, buf);
        }
        OpCode::MakeObject { num_fields } => {
            encode_uleb128_into(*num_fields as u64, buf);
        }
        OpCode::MakeArray { num_elements } => {
            encode_uleb128_into(*num_elements as u64, buf);
        }
        OpCode::MakeInteger { value } => {
            encode_uleb128_into(*value as u64, buf);
        }
        OpCode::MakeString { value } => {
            encode_uleb128_into(*value as u64, buf);
        }
        OpCode::MakeNull => {}
        OpCode::MakeBoolean { value } => {
            encode_uleb128_into(if *value { 1 } else { 0 }, buf);
        }
        OpCode::CallFunction {
            num_args,
            func_index,
        } => {
            encode_uleb128_into(*num_args as u64, buf);
            encode_uleb128_into(*func_index as u64, buf);
        }
        OpCode::StoreGlobal { var_index } => {
            encode_uleb128_into(*var_index as u64, buf);
        }
        OpCode::StoreLocal { var_index } => {
            encode_uleb128_into(*var_index as u64, buf);
        }
        OpCode::Add => {}
        OpCode::LoadGlobal { var_index } => {
            encode_uleb128_into(*var_index as u64, buf);
        }
        OpCode::LoadLocal { var_index } => {
            encode_uleb128_into(*var_index as u64, buf);
        }
        OpCode::Nop => {}
        OpCode::MakeFloat { value } => {
            encode_uleb128_into(*value as u64, buf);
        }
    }
}

pub fn encode(instr: &OpCode) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8);
    encode_into(instr, &mut buf);
    buf
}

fn encode_instrs_into(instrs: &[OpCode], buf: &mut Vec<u8>) {
    encode_uleb128_into(instrs.len() as u64, buf);

    for instr in instrs {
        encode_into(instr, buf);
    }
}

pub fn encode_instrs(instrs: &[OpCode]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(instrs.len().saturating_mul(4).saturating_add(8));
    encode_instrs_into(instrs, &mut buf);
    buf
}

fn encode_literal_into(literal: &Literal, buf: &mut Vec<u8>) {
    match literal {
        Literal::Integer(i) => {
            buf.push(0x00);
            encode_sleb128_into(*i, buf);
        }
        Literal::String(s) => {
            buf.push(0x01);
            encode_uleb128_into(s.len() as u64, buf);
            buf.extend(s.as_bytes());
        }
        Literal::Float(f) => {
            buf.push(0x02);
            buf.extend(&f.to_le_bytes());
        }
    }
}

pub fn encode_literal(literal: &Literal) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8);
    encode_literal_into(literal, &mut buf);
    buf
}

fn encode_literals_into(literals: &[Literal], buf: &mut Vec<u8>) {
    encode_uleb128_into(literals.len() as u64, buf);

    for literal in literals {
        encode_literal_into(literal, buf);
    }
}

pub fn encode_literals(literals: &[Literal]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(literals.len().saturating_mul(8).saturating_add(8));
    encode_literals_into(literals, &mut buf);
    buf
}

pub fn write_instrs(
    instrs: &[OpCode],
    literals: &[Literal],
    file_name: &str,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(file_name)?;
    let mut buf = Vec::with_capacity(
        literals
            .len()
            .saturating_mul(8)
            .saturating_add(instrs.len().saturating_mul(4))
            .saturating_add(16),
    );
    encode_literals_into(literals, &mut buf);
    encode_instrs_into(instrs, &mut buf);
    file.write_all(&buf)?;

    Ok(())
}
