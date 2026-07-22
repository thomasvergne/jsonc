use std::fmt::Debug;

pub type Address = u32;

#[derive(Clone, Eq, PartialEq, Hash)]
pub enum OpCode {
    MakeFunction {
        num_params: u8,
        body_len: u16,
    },
    MakeField {
        field_name: Address,
    },
    MakeObject {
        num_fields: u32,
    },
    MakeArray {
        num_elements: u32,
    },
    MakeInteger {
        value: Address,
    },
    MakeIntegerImm {
        value: i64,
    },
    MakeString {
        value: Address,
    },
    MakeNull,
    MakeBoolean {
        value: bool,
    },
    CallFunction {
        num_args: u8,
        func_index: Address,
    },
    StoreGlobal {
        var_index: Address,
    },
    StoreLocal {
        var_index: Address,
    },
    Add,
    LoadGlobal {
        var_index: Address,
    },
    LoadLocal {
        var_index: Address,
    },
    Nop,
    MakeFloat {
        value: Address,
    },

    // Super-instructions
    AddAddr {
        addr: Address,
    },
    AddAddr2 {
        addr1: Address,
        addr2: Address,
    },
    AddGlobalLeft {
        addr: Address,
    },
    AddGlobalRight {
        addr: Address,
    },
    AddGlobal2 {
        addr1: Address,
        addr2: Address,
    },
    MakeFieldFromAddr {
        field_name: Address,
        addr: Address,
    },
    MakeFieldFromGlobal {
        field_name: Address,
        addr: Address,
    },
    MakeFieldFromLocal {
        field_name: Address,
        addr: Address,
    },
    MakeArray2 {
        addr1: Address,
        addr2: Address,
    },
    MakeFieldFromIntegerImm {
        field_name: Address,
        value: i64,
    },
    MakeFieldFromBoolean {
        field_name: Address,
        value: bool,
    },
    MakePairArray {
        pairs: Vec<(Address, Address)>,
    },
    MakeFieldIntegerBlock {
        fields: Vec<(Address, i64)>,
    },
    MakeFieldIntegerBlockSequential {
        count: u32,
        start_value: i64,
        start_field_address: Address,
    },
    MakeFieldGlobalBlock {
        fields: Vec<(Address, Address)>,
    },
    MakeFieldLocalBlock {
        fields: Vec<(Address, Address)>,
    },
    MakeFieldAddrBlock {
        fields: Vec<(Address, Address)>,
    },
    MakeFieldBooleanBlock {
        fields: Vec<(Address, bool)>,
    },
    MakeStringInline {
        value: String,
    },
    MakeStringNumInline {
        value: u64,
    },
    MakeStringTsInline {
        value: u64,
    },
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum Literal {
    Integer(i64),
    Float(u64),
    String(String),
    Null,
    Bool(bool),
}

impl OpCode {
    pub fn opcode(&self) -> u8 {
        match self {
            OpCode::MakeFunction { .. } => 0,
            OpCode::MakeField { .. } => 1,
            OpCode::MakeObject { .. } => 2,
            OpCode::MakeArray { .. } => 3,
            OpCode::MakeInteger { .. } => 4,
            OpCode::MakeIntegerImm { .. } => 0x52,
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
            OpCode::AddAddr { .. } => 16,
            OpCode::AddAddr2 { .. } => 17,
            OpCode::MakeFieldFromAddr { .. } => 18,
            OpCode::AddGlobalLeft { .. } => 19,
            OpCode::AddGlobalRight { .. } => 31,
            OpCode::AddGlobal2 { .. } => 20,
            OpCode::MakeFieldFromGlobal { .. } => 21,
            OpCode::MakeArray2 { .. } => 22,
            OpCode::MakeFieldFromIntegerImm { .. } => 23,
            OpCode::MakeFieldFromBoolean { .. } => 24,
            OpCode::MakePairArray { .. } => 25,
            OpCode::MakeFieldIntegerBlock { .. } => 26,
            OpCode::MakeFieldIntegerBlockSequential { .. } => 27,
            OpCode::MakeFieldGlobalBlock { .. } => 28,
            OpCode::MakeFieldAddrBlock { .. } => 29,
            OpCode::MakeFieldBooleanBlock { .. } => 30,
            OpCode::MakeStringInline { .. } => 99,
            OpCode::MakeStringNumInline { .. } => 100,
            OpCode::MakeStringTsInline { .. } => 101,
            OpCode::MakeFieldLocalBlock { .. } => 102,
            OpCode::MakeFieldFromLocal { .. } => 103,
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
            OpCode::MakeIntegerImm { value } => write!(f, "make_integer_imm value={}", value),
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
            OpCode::AddAddr { addr } => write!(f, "add_addr addr={}", addr),
            OpCode::AddAddr2 { addr1, addr2 } => {
                write!(f, "add_addr2 addr1={} addr2={}", addr1, addr2)
            }
            OpCode::MakeFieldFromAddr { field_name, addr } => {
                write!(
                    f,
                    "make_field_from_addr field_name={} addr={}",
                    field_name, addr
                )
            }
            OpCode::AddGlobalLeft { addr } => write!(f, "add_global_left addr={}", addr),
            OpCode::AddGlobalRight { addr } => write!(f, "add_global_right addr={}", addr),
            OpCode::AddGlobal2 { addr1, addr2 } => {
                write!(f, "add_global2 addr1={} addr2={}", addr1, addr2)
            }
            OpCode::MakeFieldFromGlobal { field_name, addr } => {
                write!(
                    f,
                    "make_field_from_global field_name={} addr={}",
                    field_name, addr
                )
            }
            OpCode::MakeArray2 { addr1, addr2 } => {
                write!(f, "make_array2 addr1={} addr2={}", addr1, addr2)
            }
            OpCode::MakeFieldFromIntegerImm { field_name, value } => {
                write!(
                    f,
                    "make_field_from_integer_imm field_name={} value={}",
                    field_name, value
                )
            }
            OpCode::MakeFieldFromBoolean { field_name, value } => {
                write!(
                    f,
                    "make_field_from_boolean field_name={} value={}",
                    field_name, value
                )
            }
            OpCode::MakePairArray { pairs } => {
                write!(f, "make_pair_array len={}", pairs.len())
            }
            OpCode::MakeFieldIntegerBlock { fields } => {
                write!(f, "make_field_integer_block len={}", fields.len())
            }
            OpCode::MakeFieldIntegerBlockSequential {
                count,
                start_value,
                start_field_address,
            } => {
                write!(
                    f,
                    "make_field_integer_block_sequential count={} start_value={} start_field_address={}",
                    count, start_value, start_field_address
                )
            }
            OpCode::MakeFieldGlobalBlock { fields } => {
                write!(f, "make_field_global_block len={}", fields.len())
            }
            OpCode::MakeFieldAddrBlock { fields } => {
                write!(f, "make_field_addr_block len={}", fields.len())
            }
            OpCode::MakeFieldBooleanBlock { fields } => {
                write!(f, "make_field_boolean_block len={}", fields.len())
            }
            OpCode::MakeStringInline { value } => {
                write!(f, "make_string_inline value={:?}", value)
            }
            OpCode::MakeStringNumInline { value } => {
                write!(f, "make_string_num_inline value={}", value)
            }
            OpCode::MakeStringTsInline { value } => {
                write!(f, "make_string_ts_inline value={}", value)
            }
            OpCode::MakeFieldLocalBlock { fields } => {
                write!(f, "make_field_local_block len={}", fields.len())
            }
            OpCode::MakeFieldFromLocal { field_name, addr } => {
                write!(
                    f,
                    "make_field_from_local field_name={} addr={}",
                    field_name, addr
                )
            }
        }
    }
}
