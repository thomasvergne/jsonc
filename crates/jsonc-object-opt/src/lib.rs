use std::collections::HashMap;

use jsonc_mlir::MLIR;

#[derive(Clone, Debug)]
pub struct ObjectOptimizer<'a> {
    pub objects: Vec<MLIR<'a>>,
    pub new_functions: HashMap<&'a str, MLIR<'a>>,
}

pub fn replace_every<'a>(mlir: MLIR<'a>, value: &(&'a str, MLIR<'a>)) -> MLIR<'a> {
    match mlir {
        mlir if &mlir == &value.1 => MLIR::Variable(value.0),

        MLIR::Object(hm) => MLIR::Object(
            hm.into_iter()
                .map(|(k, v)| (k, replace_every(v, value)))
                .collect(),
        ),

        MLIR::Array(elements) => MLIR::Array(
            elements
                .into_iter()
                .map(|v| replace_every(v, value))
                .collect(),
        ),

        MLIR::Add { left, right } => MLIR::Add {
            left: Box::new(replace_every(*left, value)),
            right: Box::new(replace_every(*right, value)),
        },

        MLIR::FunctionCall { name, args } => MLIR::FunctionCall {
            name,
            args: args.into_iter().map(|v| replace_every(v, value)).collect(),
        },

        MLIR::MakeFunction { name, params, body } => MLIR::MakeFunction {
            name,
            params,
            body: Box::new(replace_every(*body, value)),
        },

        MLIR::Let {
            name,
            value: new_value,
        } => MLIR::Let {
            name,
            value: Box::new(replace_every(*new_value, value)),
        },

        _ => mlir,
    }
}

pub fn replace_every_multiple<'a>(mlir: MLIR<'a>, values: &[(&'a str, MLIR<'a>)]) -> MLIR<'a> {
    values
        .iter()
        .fold(mlir, |acc, value| replace_every(acc, value))
}

/// Remplace chaque nœud `Variable(name)` par `val`.
///
/// `val` est passé par référence : le clone n'a lieu qu'au moment de la
/// substitution effective, évitant O(pool_size) clones par site d'appel.
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
                // Consomme la String directement, sans clone intermédiaire
                let function_str: &'static str = Box::leak(keys.join("_").into_boxed_str());

                let pairs: Vec<(&str, MLIR<'a>)> = hm
                    .into_iter()
                    .map(|(k, v)| (k, self.optimize(v, false)))
                    .collect();

                let result = MLIR::Object(pairs.clone());

                if !root && self.new_functions.get(function_str).is_none() {
                    self.add_json(result.clone());
                    let body = Box::new(replace_every_multiple(result.clone(), &pairs));
                    self.new_functions.insert(
                        function_str,
                        MLIR::MakeFunction {
                            name: function_str,
                            params: keys.clone(),
                            body, // move au lieu de clone
                        },
                    );
                }

                if !root {
                    if let Some(existing) = self.new_functions.get(function_str) {
                        let optimized_existing = self.optimize(existing.clone(), false);
                        self.new_functions
                            .insert(function_str, optimized_existing.clone());

                        return MLIR::FunctionCall {
                            name: function_str,
                            args: keys
                                .iter()
                                .map(|k| {
                                    pairs
                                        .iter()
                                        .find(|(k_, _)| k == k_)
                                        .map(|(_, v)| v.clone())
                                        .unwrap_or(MLIR::Null)
                                })
                                .collect(),
                        };
                    }
                }

                result
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
