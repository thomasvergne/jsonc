use std::collections::HashMap;

use jsonc_bytecode::{Address, Literal, OpCode};
use jsonc_mlir::MLIR;

pub struct Compiler {
    pub value_pool: Vec<Literal>,
    literal_indices: HashMap<Literal, Address>,
    locals: Vec<String>,
    local_indices: HashMap<String, Address>,
    globals: Vec<String>,
    global_indices: HashMap<String, Address>,
    pub functions: HashMap<String, Address>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            value_pool: Vec::new(),
            literal_indices: HashMap::new(),
            locals: Vec::new(),
            local_indices: HashMap::new(),
            globals: Vec::new(),
            global_indices: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    fn intern_literal(&mut self, literal: Literal) -> Address {
        if let Some(index) = self.literal_indices.get(&literal) {
            return *index;
        }

        let index = self.value_pool.len() as Address;
        self.value_pool.push(literal.clone());
        self.literal_indices.insert(literal, index);
        index
    }

    fn push_local(&mut self, name: &str) {
        if self.local_indices.contains_key(name) {
            return;
        }
        let index = self.locals.len() as Address;
        self.locals.push(name.to_string());
        self.local_indices.insert(name.to_string(), index);
    }

    fn push_global(&mut self, name: &str) {
        if self.global_indices.contains_key(name) {
            return;
        }
        let index = self.globals.len() as Address;
        self.globals.push(name.to_string());
        self.global_indices.insert(name.to_string(), index);
    }

    pub fn compile_all(&mut self, mlir: Vec<MLIR<'_>>) -> Vec<OpCode> {
        let mut instructions = Vec::with_capacity(mlir.len() * 4);

        for node in &mlir {
            instructions.extend(self.compile(&node, false));
        }

        instructions
    }

    fn compile_into<'a>(
        &mut self,
        mlir: &MLIR<'a>,
        is_in_function: bool,
        instructions: &mut Vec<OpCode>,
    ) {
        match mlir {
            MLIR::String(s) => {
                let index = self.intern_literal(Literal::String(s.to_string()));
                instructions.push(OpCode::MakeString { value: index });
            }

            MLIR::Number(num) => {
                if num.fract() == 0.0 && *num >= (i64::MIN as f64) && *num <= (i64::MAX as f64) {
                    let index = self.intern_literal(Literal::Integer(*num as i64));
                    instructions.push(OpCode::MakeInteger { value: index });
                } else {
                    let index = self.intern_literal(Literal::Float(num.to_bits()));
                    instructions.push(OpCode::MakeFloat { value: index });
                }
            }

            MLIR::Null => {
                instructions.push(OpCode::MakeNull);
            }

            MLIR::Bool(b) => {
                instructions.push(OpCode::MakeBoolean { value: *b });
            }

            MLIR::Array(arr) => {
                let len = arr.len();
                for item in arr {
                    self.compile_into(item, is_in_function, instructions);
                }
                instructions.push(OpCode::MakeArray {
                    num_elements: len as u32,
                });
            }

            MLIR::Object(obj) => {
                for (key, value) in obj {
                    let index = self.intern_literal(Literal::String(key.to_string()));
                    self.compile_into(value, is_in_function, instructions);

                    instructions.push(OpCode::MakeField { field_name: index });
                }
                instructions.push(OpCode::MakeObject {
                    num_fields: obj.len() as u32,
                });
            }

            MLIR::Variable(name) => {
                if let Some(index) = self.local_indices.get(*name) {
                    instructions.push(OpCode::LoadLocal { var_index: *index });
                } else if let Some(index) = self.global_indices.get(*name) {
                    instructions.push(OpCode::LoadGlobal { var_index: *index });
                }
            }

            MLIR::MakeFunction { name, params, body } => {
                let old_locals = self.locals.clone();
                let old_local_indices = self.local_indices.clone();

                for param in params {
                    self.push_local(param);
                }

                let header_index = instructions.len();
                instructions.push(OpCode::MakeFunction {
                    num_params: params.len() as u8,
                    body_len: 0,
                });

                self.functions
                    .insert(name.to_string(), instructions.len() as Address);
                let body_start = instructions.len();
                self.compile_into(body, true, instructions);
                let body_len = (instructions.len() - body_start) as u16;

                if let OpCode::MakeFunction {
                    body_len: header_body_len,
                    ..
                } = &mut instructions[header_index]
                {
                    *header_body_len = body_len;
                }

                self.locals = old_locals;
                self.local_indices = old_local_indices;
                self.push_global(name);
            }

            MLIR::FunctionCall { name, args } => {
                let func_index = *self.functions.get(*name).expect(&format!(
                    "FunctionCall references an unknown function: {}",
                    name
                ));
                for arg in args {
                    self.compile_into(arg, is_in_function, instructions);
                }
                instructions.push(OpCode::CallFunction {
                    num_args: args.len() as u8,
                    func_index,
                });
            }

            MLIR::Add { left, right } => {
                self.compile_into(left, is_in_function, instructions);
                self.compile_into(right, is_in_function, instructions);
                instructions.push(OpCode::Add);
            }

            MLIR::Let { name, value } => {
                self.compile_into(value, is_in_function, instructions);

                let index = if is_in_function {
                    self.locals.len() as Address
                } else {
                    self.globals.len() as Address
                };

                instructions.push(if is_in_function {
                    OpCode::StoreLocal { var_index: index }
                } else {
                    OpCode::StoreGlobal { var_index: index }
                });

                if is_in_function {
                    self.push_local(name);
                } else {
                    self.push_global(name);
                }
            }
        }
    }

    pub fn compile<'a>(&mut self, mlir: &MLIR<'a>, is_in_function: bool) -> Vec<OpCode> {
        let mut instructions = Vec::new();
        self.compile_into(mlir, is_in_function, &mut instructions);

        instructions
    }
}
