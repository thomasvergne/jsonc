use std::fmt::Debug;

pub type Address = u32;

#[derive(Clone, Copy)]
pub enum OpCode {
    MakeFunction { num_params: u8, body_len: u16 },
    MakeField { field_name: Address },
    MakeObject { num_fields: u32 },
    MakeArray { num_elements: u32 },
    MakeInteger { value: Address },
    MakeString { value: Address },
    MakeNull,
    MakeBoolean { value: bool },
    CallFunction { num_args: u8, func_index: Address },
    StoreGlobal { var_index: Address },
    StoreLocal { var_index: Address },
    Add,
    LoadGlobal { var_index: Address },
    LoadLocal { var_index: Address },
    Nop,
    MakeFloat { value: Address },
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum Literal {
    Integer(i64),
    Float(u64),
    String(String),
}

impl OpCode {
    pub fn opcode(&self) -> u8 {
        match self {
            OpCode::MakeFunction { .. } => 0,
            OpCode::MakeField { .. } => 1,
            OpCode::MakeObject { .. } => 2,
            OpCode::MakeArray { .. } => 3,
            OpCode::MakeInteger { .. } => 4,
            OpCode::MakeString { .. } => 5,
            OpCode::MakeNull => 6,
            OpCode::MakeBoolean { .. } => 7,
            OpCode::CallFunction { .. } => 8,
            OpCode::StoreGlobal { .. } => 9,
            OpCode::StoreLocal { .. } => 10,
            OpCode::Add => 11,
            OpCode::LoadGlobal { .. } => 12,
            OpCode::LoadLocal { .. } => 13,
            OpCode::Nop => 14,
            OpCode::MakeFloat { .. } => 15,
        }
    }
}

impl Debug for OpCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpCode::MakeFunction {
                num_params,
                body_len,
            } => write!(
                f,
                "make_function num_params={} body_len={}",
                num_params, body_len
            ),
            OpCode::MakeField { field_name } => write!(f, "make_field field_name={}", field_name),
            OpCode::MakeObject { num_fields } => write!(f, "make_object num_fields={}", num_fields),
            OpCode::MakeArray { num_elements } => {
                write!(f, "make_array num_elements={}", num_elements)
            }
            OpCode::MakeInteger { value } => write!(f, "make_integer value={}", value),
            OpCode::MakeFloat { value } => write!(f, "make_float value={}", value),
            OpCode::MakeString { value } => write!(f, "make_string value={}", value),
            OpCode::MakeNull => write!(f, "make_null"),
            OpCode::MakeBoolean { value } => write!(f, "make_boolean value={}", value),
            OpCode::CallFunction {
                num_args,
                func_index,
            } => write!(
                f,
                "call_function num_args={} func_index={}",
                num_args, func_index
            ),
            OpCode::StoreGlobal { var_index } => write!(f, "store_global var_index={}", var_index),
            OpCode::StoreLocal { var_index } => write!(f, "store_local var_index={}", var_index),
            OpCode::Add => write!(f, "add"),
            OpCode::LoadGlobal { var_index } => write!(f, "load_global var_index={}", var_index),
            OpCode::LoadLocal { var_index } => write!(f, "load_local var_index={}", var_index),
            OpCode::Nop => write!(f, "nop"),
        }
    }
}
