use std::collections::HashMap;

use jsonc_bytecode::{Address, Literal, OpCode};
use jsonc_mlir::MLIR;

pub struct Compiler {
    pub value_pool: Vec<Literal>,
    locals: Vec<String>,
    globals: Vec<String>,
    pub functions: HashMap<String, usize>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            value_pool: Vec::new(),
            locals: Vec::new(),
            globals: Vec::new(),
            functions: HashMap::new(),
        }
    }

    pub fn compile_all(&mut self, mlir: Vec<MLIR<'_>>) -> Vec<OpCode> {
        let mut instructions = Vec::new();

        for node in &mlir {
            instructions.extend(self.compile(&node, false));
        }

        instructions
    }

    pub fn compile<'a>(&mut self, mlir: &MLIR<'a>, is_in_function: bool) -> Vec<OpCode> {
        let mut instructions = Vec::new();

        match mlir {
            MLIR::String(s) => {
                let index = self.value_pool.len();
                self.value_pool.push(Literal::String(s.to_string()));
                instructions.push(OpCode::MakeString {
                    value: index as Address,
                });
            }

            MLIR::Number(num) => {
                let index = self.value_pool.len();
                self.value_pool.push(Literal::Integer(*num as i64));
                instructions.push(OpCode::MakeInteger {
                    value: index as Address,
                });
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
                    instructions.extend(self.compile(item, is_in_function));
                }
                instructions.push(OpCode::MakeArray {
                    num_elements: len as u32,
                });
            }

            MLIR::Object(obj) => {
                for (key, value) in obj {
                    // Avoid allocating a temporary Literal for each position check.
                    let index = if let Some((idx, _)) =
                        self.value_pool
                            .iter()
                            .enumerate()
                            .find(|(_, lit)| match lit {
                                Literal::String(existing) => existing.as_str() == *key,
                                _ => false,
                            }) {
                        idx as u8
                    } else {
                        let index = self.value_pool.len();
                        self.value_pool.push(Literal::String(key.to_string()));
                        index as u8
                    };

                    instructions.extend(self.compile(value, is_in_function));

                    instructions.push(OpCode::MakeField {
                        field_name: index as Address,
                    });
                }
                instructions.push(OpCode::MakeObject {
                    num_fields: obj.len() as u32,
                });
            }

            MLIR::Variable(name) => {
                if let Some(index) = self.locals.iter().position(|n| n == name) {
                    instructions.push(OpCode::LoadLocal {
                        var_index: index as Address,
                    });
                } else if let Some(index) = self.globals.iter().position(|n| n == name) {
                    instructions.push(OpCode::LoadGlobal {
                        var_index: index as Address,
                    });
                }
            }

            MLIR::MakeFunction { name, params, body } => {
                let old_locals = self.locals.clone();
                self.locals.extend(params.iter().map(|p| p.to_string()));

                let body_instructions = self.compile(body, true);

                self.locals = old_locals;

                instructions.push(OpCode::MakeFunction {
                    num_params: params.len() as u8,
                    body_len: body_instructions.len() as u16,
                });

                self.functions.insert(name.to_string(), instructions.len());
                instructions.extend(body_instructions);
                self.globals.push(name.to_string());
            }

            MLIR::FunctionCall { name, args } => {
                let func_index = self.functions[*name] as Address;

                // collect arg instructions, but avoid cloning the collected Vec
                let args_instructions: Vec<OpCode> = args
                    .iter()
                    .flat_map(|arg| self.compile(arg, is_in_function))
                    .collect();
                let args_len = args_instructions.len();
                instructions.extend(args_instructions);
                instructions.push(OpCode::CallFunction {
                    num_args: args_len as u8,
                    func_index,
                });
            }

            MLIR::Add { left, right } => {
                let left_instructions = self.compile(left, is_in_function);
                let right_instructions = self.compile(right, is_in_function);

                instructions.extend(left_instructions);
                instructions.extend(right_instructions);
                instructions.push(OpCode::Add);
            }

            MLIR::Let { name, value } => {
                let value_instructions = self.compile(value, is_in_function);

                let index = if is_in_function {
                    self.locals.len() as Address
                } else {
                    self.globals.len() as Address
                };

                instructions.extend(value_instructions);
                instructions.push(if is_in_function {
                    OpCode::StoreLocal { var_index: index }
                } else {
                    OpCode::StoreGlobal { var_index: index }
                });

                if is_in_function {
                    self.locals.push(name.to_string());
                } else {
                    self.globals.push(name.to_string());
                }
            }
        }

        instructions
    }
}
