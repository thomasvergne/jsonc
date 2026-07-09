use std::collections::HashMap;

use jsonc_mlir::MLIR;

#[derive(Clone, Debug)]
pub struct ObjectOptimizer<'a> {
    pub objects: Vec<MLIR<'a>>,
    pub new_functions: HashMap<&'a str, MLIR<'a>>,
}

pub fn replace_multiple_variables_with_values<'a>(
    mlir: MLIR<'a>,
    replacements: &HashMap<&'a str, MLIR<'a>>,
) -> MLIR<'a> {
    match mlir {
        MLIR::Variable(n) => {
            if let Some(val) = replacements.get(n) {
                val.clone()
            } else {
                MLIR::Variable(n)
            }
        }

        MLIR::Object(hm) => MLIR::Object(
            hm.into_iter()
                .map(|(k, v)| (k, replace_multiple_variables_with_values(v, replacements)))
                .collect(),
        ),

        MLIR::Array(elements) => MLIR::Array(
            elements
                .into_iter()
                .map(|v| replace_multiple_variables_with_values(v, replacements))
                .collect(),
        ),

        MLIR::Add { left, right } => MLIR::Add {
            left: Box::new(replace_multiple_variables_with_values(*left, replacements)),
            right: Box::new(replace_multiple_variables_with_values(*right, replacements)),
        },

        MLIR::FunctionCall {
            name: fn_name,
            args,
        } => MLIR::FunctionCall {
            name: fn_name,
            args: args
                .into_iter()
                .map(|v| replace_multiple_variables_with_values(v, replacements))
                .collect(),
        },

        MLIR::MakeFunction {
            name: fn_name,
            params,
            body,
        } => MLIR::MakeFunction {
            name: fn_name,
            params,
            body: Box::new(replace_multiple_variables_with_values(*body, replacements)),
        },

        MLIR::Let {
            name: let_name,
            value,
        } => MLIR::Let {
            name: let_name,
            value: Box::new(replace_multiple_variables_with_values(*value, replacements)),
        },

        _ => mlir,
    }
}

/// Replace every node `Variable(name)` by `val`.
pub fn replace_variable_with_value<'a>(mlir: &MLIR<'a>, name: &'a str, val: &MLIR<'a>) -> MLIR<'a> {
    match mlir {
        MLIR::Variable(n) if *n == name => val.clone(),

        MLIR::Object(hm) => MLIR::Object(
            hm.iter()
                .map(|(k, v)| (*k, replace_variable_with_value(v, name, val)))
                .collect(),
        ),

        MLIR::Array(elements) => MLIR::Array(
            elements
                .iter()
                .map(|v| replace_variable_with_value(v, name, val))
                .collect(),
        ),

        MLIR::Add { left, right } => MLIR::Add {
            left: Box::new(replace_variable_with_value(left, name, val)),
            right: Box::new(replace_variable_with_value(right, name, val)),
        },

        MLIR::FunctionCall {
            name: fn_name,
            args,
        } => MLIR::FunctionCall {
            name: *fn_name,
            args: args
                .iter()
                .map(|v| replace_variable_with_value(v, name, val))
                .collect(),
        },

        MLIR::MakeFunction {
            name: fn_name,
            params,
            body,
        } => MLIR::MakeFunction {
            name: *fn_name,
            params: params.clone(),
            body: Box::new(replace_variable_with_value(body, name, val)),
        },

        MLIR::Let {
            name: let_name,
            value,
        } => MLIR::Let {
            name: *let_name,
            value: Box::new(replace_variable_with_value(value, name, val)),
        },

        _ => mlir.clone(),
    }
}

impl<'a> ObjectOptimizer<'a> {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            new_functions: HashMap::new(),
        }
    }

    pub fn add_json(&mut self, mlir: MLIR<'a>) {
        self.objects.push(mlir);
    }

    pub fn find_similar_objects(&mut self, mlir: MLIR<'a>) -> Vec<&'a str> {
        let mut result = Vec::new();
        for object in &self.objects {
            if let MLIR::Object(hm1) = object
                && let MLIR::Object(hm2) = &mlir
                && hashmap_keys(hm1) == hashmap_keys(hm2)
            {
                result.extend(hashmap_keys(hm1));
                return result;
            }
        }
        result
    }

    pub fn format_output(&self, mlir: MLIR<'a>) -> Vec<MLIR<'a>> {
        let mut result = self.new_functions.values().cloned().collect::<Vec<_>>();
        result.push(mlir);
        result
    }

    pub fn optimize(&mut self, mlir: MLIR<'a>, root: bool) -> MLIR<'a> {
        match mlir {
            MLIR::Object(hm) => {
                let keys: Vec<&str> = hm.iter().map(|(k, _)| *k).collect();
                let function_str: &'static str = Box::leak(keys.join("_").into_boxed_str());

                let pairs: Vec<(&str, MLIR<'a>)> = hm
                    .into_iter()
                    .map(|(k, v)| (k, self.optimize(v, false)))
                    .collect();

                if !root && !self.new_functions.contains_key(function_str) {
                    self.add_json(MLIR::Object(pairs.clone()));
                    let body = Box::new(MLIR::Object(
                        keys.iter().map(|&k| (k, MLIR::Variable(k))).collect(),
                    ));
                    self.new_functions.insert(
                        function_str,
                        MLIR::MakeFunction {
                            name: function_str,
                            params: keys.clone(),
                            body,
                        },
                    );
                }

                if !root {
                    let args = pairs.into_iter().map(|(_, v)| v).collect();
                    return MLIR::FunctionCall {
                        name: function_str,
                        args,
                    };
                }

                MLIR::Object(pairs)
            }

            MLIR::Array(vec) => {
                MLIR::Array(vec.into_iter().map(|v| self.optimize(v, false)).collect())
            }

            MLIR::Add { left, right } => MLIR::Add {
                left: Box::new(self.optimize(*left, false)),
                right: Box::new(self.optimize(*right, false)),
            },

            MLIR::MakeFunction { name, params, body } => MLIR::MakeFunction { name, params, body },

            MLIR::FunctionCall { name, args } => MLIR::FunctionCall {
                name,
                args: args.into_iter().map(|a| self.optimize(a, false)).collect(),
            },

            MLIR::Let { name, value } => MLIR::Let {
                name,
                value: Box::new(self.optimize(*value, false)),
            },

            _ => mlir,
        }
    }
}

pub const IS_ROOT: bool = true;

pub fn object_len<T>(hm: &Vec<(&str, T)>) -> usize {
    hm.len()
}

pub fn hashmap_keys<'a, T>(hm: &Vec<(&'a str, T)>) -> Vec<&'a str> {
    let mut keys: Vec<&'a str> = hm.iter().map(|(k, _)| *k).collect();
    keys.sort_unstable();
    keys
}
