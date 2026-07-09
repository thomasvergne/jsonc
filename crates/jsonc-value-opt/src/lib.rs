use std::collections::{HashMap, HashSet};

use jsonc_mlir::{MLIR, debug};
use jsonc_object_opt::{
    IS_ROOT, replace_multiple_variables_with_values, replace_variable_with_value,
};

pub struct ValueOptimizer<'a> {
    pub value_pool: Vec<MLIR<'a>>,
    pub value_counter: HashMap<usize, usize>,
    pub value_to_drop: HashSet<usize>,
    var_name_cache: Vec<&'static str>,
    value_indices: HashMap<MLIR<'a>, usize>,
}

impl<'a> ValueOptimizer<'a> {
    pub fn new() -> Self {
        Self {
            value_pool: Vec::new(),
            value_counter: HashMap::new(),
            value_to_drop: HashSet::new(),
            var_name_cache: Vec::new(),
            value_indices: HashMap::new(),
        }
    }

    /// Returns and caches the static string "v{index}".
    fn var_name(&mut self, index: usize) -> &'static str {
        if index < self.var_name_cache.len() {
            return self.var_name_cache[index];
        }
        while self.var_name_cache.len() <= index {
            let i = self.var_name_cache.len();
            let s: &'static str = Box::leak(format!("v{}", i).into_boxed_str());
            self.var_name_cache.push(s);
        }
        self.var_name_cache[index]
    }

    pub fn create_variable(&mut self, index: usize) -> MLIR<'a> {
        MLIR::Variable(self.var_name(index))
    }

    pub fn create_lets(&mut self) -> Vec<MLIR<'a>> {
        let entries: Vec<(usize, MLIR<'a>)> = self
            .value_pool
            .iter()
            .enumerate()
            .filter(|(index, value)| {
                !matches!(value, MLIR::MakeFunction { .. }) && !self.value_to_drop.contains(index)
            })
            .map(|(index, value)| (index, value.clone()))
            .collect();

        let mut lets = Vec::new();
        for (index, value) in entries {
            lets.push(MLIR::Let {
                name: self.var_name(index),
                value: Box::new(value),
            });
        }
        self.reanalyze_lets(lets)
    }

    pub fn reanalyze_lets(&mut self, lets: Vec<MLIR<'a>>) -> Vec<MLIR<'a>> {
        let mut new_lets = Vec::new();
        for let_ in lets.iter() {
            let value = self.optimize(let_, false);
            match value {
                MLIR::MakeFunction { .. } | MLIR::Variable(_) => {}
                v => new_lets.push(v),
            }
        }
        new_lets
    }

    pub fn collect(&mut self, mlir: &MLIR<'a>) -> MLIR<'a> {
        self.optimize(mlir, IS_ROOT)
    }

    pub fn optimize_all(&mut self, mlirs: &Vec<MLIR<'a>>) -> Vec<MLIR<'a>> {
        let mut result = Vec::new();
        for mlir in mlirs.iter() {
            debug!("Collecting values...");
            result.push(self.collect(mlir));
        }
        result
    }

    pub fn optimize_program(&mut self, mut formatted_mlir: Vec<MLIR<'a>>) -> Vec<MLIR<'a>> {
        let mut lets = self.value_pool.clone();
        let pool_len = lets.len();

        let mut resolved_replacements = HashMap::new();
        let mut resolved_functions = HashMap::new();

        debug!("Resolving lets sequentially...");
        for index in 0..pool_len {
            let count = *self.value_counter.get(&index).unwrap_or(&0);

            let old = std::mem::replace(&mut lets[index], MLIR::Null);
            let val = replace_multiple_variables_with_values(old, &resolved_replacements);
            let resolved_let = replace_multiple_every_function_call_in(val, &resolved_functions);

            if count <= 1 {
                self.value_to_drop.insert(index);
                match &resolved_let {
                    MLIR::MakeFunction { name, params, body } => {
                        resolved_functions.insert(*name, (*body.clone(), params.clone()));
                    }
                    let_value => {
                        resolved_replacements.insert(self.var_name(index), let_value.clone());
                    }
                }
                self.value_counter.remove(&index);
                lets[index] = resolved_let;
            } else {
                lets[index] = resolved_let;
            }
        }

        for _ in 0..5 {
            let mut updated = HashMap::new();
            for (name, (body, params)) in resolved_functions.iter() {
                let resolved_body =
                    replace_multiple_every_function_call_in(body.clone(), &resolved_functions);
                updated.insert(*name, (resolved_body, params.clone()));
            }
            resolved_functions = updated;
        }

        for val in lets.iter_mut() {
            let old = std::mem::replace(val, MLIR::Null);
            *val = replace_multiple_every_function_call_in(old, &resolved_functions);
        }

        self.value_pool = lets;

        for mlir in formatted_mlir.iter_mut() {
            let old = std::mem::replace(mlir, MLIR::Null);
            let val = replace_multiple_variables_with_values(old, &resolved_replacements);
            *mlir = replace_multiple_every_function_call_in(val, &resolved_functions);
        }

        formatted_mlir
    }

    pub fn remove_unused_variables(&mut self, mlirs: Vec<MLIR<'a>>) -> Vec<MLIR<'a>> {
        if mlirs.is_empty() {
            return mlirs;
        }
        let last = mlirs.len() - 1;
        let mut result = Vec::with_capacity(mlirs.len());
        for mlir in mlirs[..last].iter() {
            match mlir {
                MLIR::Variable(_) | MLIR::FunctionCall { .. } => {}
                _ => result.push(mlir.clone()),
            }
        }

        result.push(mlirs[last].clone());

        result
    }

    pub fn optimize(&mut self, mlir: &MLIR<'a>, root: bool) -> MLIR<'a> {
        match mlir {
            MLIR::String(_) | MLIR::Bool(_) | MLIR::Number(_) if !root => {
                if let Some(&index) = self.value_indices.get(mlir) {
                    *self.value_counter.entry(index).or_insert(0) += 1;
                    self.create_variable(index)
                } else {
                    let index = self.value_pool.len();
                    self.value_pool.push(mlir.clone());
                    self.value_indices.insert(mlir.clone(), index);
                    self.value_counter.insert(index, 1);
                    self.create_variable(index)
                }
            }

            MLIR::Array(elements) if !root => {
                let unique_elements = elements.iter().map(|e| self.optimize(e, false)).collect();
                let array_mlir = MLIR::Array(unique_elements);
                if let Some(&index) = self.value_indices.get(&array_mlir) {
                    *self.value_counter.entry(index).or_insert(0) += 1;
                    self.create_variable(index)
                } else {
                    let index = self.value_pool.len();
                    self.value_counter.insert(index, 1);
                    self.value_indices.insert(array_mlir.clone(), index);
                    self.value_pool.push(array_mlir);
                    self.create_variable(index)
                }
            }

            MLIR::FunctionCall { name, args } => {
                let optimized_args = args.iter().map(|a| self.optimize(a, false)).collect();
                MLIR::FunctionCall {
                    name: *name,
                    args: optimized_args,
                }
            }

            MLIR::MakeFunction { name, params, body } => {
                let optimized_body = Box::new(self.optimize(body, false));
                let function = MLIR::MakeFunction {
                    name: *name,
                    params: params.clone(),
                    body: optimized_body,
                };
                if let Some(&index) = self.value_indices.get(&function) {
                    *self.value_counter.entry(index).or_insert(0) += 1;
                } else {
                    let index = self.value_pool.len();
                    self.value_counter.insert(index, 1);
                    self.value_indices.insert(function.clone(), index);
                    self.value_pool.push(function);
                }
                MLIR::Variable(*name)
            }

            MLIR::Add { left, right } => MLIR::Add {
                left: Box::new(self.optimize(left, false)),
                right: Box::new(self.optimize(right, false)),
            },

            MLIR::Let { name, value } => {
                let optimized_value = self.optimize(value, false);
                if MLIR::Variable(name) == optimized_value {
                    return MLIR::Let {
                        name: *name,
                        value: value.clone(),
                    };
                }
                MLIR::Let {
                    name: *name,
                    value: Box::new(optimized_value),
                }
            }

            MLIR::Object(obj) => {
                let optimized_obj = obj
                    .iter()
                    .map(|(k, v)| (*k, self.optimize(v, false)))
                    .collect();
                MLIR::Object(optimized_obj)
            }

            MLIR::Array(elements) => {
                let optimized_elements = elements.iter().map(|e| self.optimize(e, false)).collect();
                MLIR::Array(optimized_elements)
            }

            _ => mlir.clone(),
        }
    }
}

pub fn replace_multiple_every_function_call_in<'a>(
    mlir: MLIR<'a>,
    functions: &HashMap<&'a str, (MLIR<'a>, Vec<&'a str>)>,
) -> MLIR<'a> {
    match mlir {
        MLIR::FunctionCall {
            name: func_name,
            args,
        } => {
            if functions.contains_key(func_name) {
                let (new_value, new_params) = &functions[func_name];
                let bound: HashMap<&'a str, MLIR<'a>> = new_params
                    .iter()
                    .zip(args.iter())
                    .map(|(p, a)| (*p, a.clone()))
                    .collect();
                let result = replace_multiple_variables_with_values(new_value.clone(), &bound);
                replace_multiple_every_function_call_in(result, functions)
            } else {
                MLIR::FunctionCall {
                    name: func_name,
                    args: args
                        .into_iter()
                        .map(|e| replace_multiple_every_function_call_in(e, functions))
                        .collect(),
                }
            }
        }

        MLIR::Add { left, right } => MLIR::Add {
            left: Box::new(replace_multiple_every_function_call_in(*left, functions)),
            right: Box::new(replace_multiple_every_function_call_in(*right, functions)),
        },

        MLIR::Array(elements) => MLIR::Array(
            elements
                .into_iter()
                .map(|e| replace_multiple_every_function_call_in(e, functions))
                .collect(),
        ),

        MLIR::Object(hm) => MLIR::Object(
            hm.into_iter()
                .map(|(k, v)| (k, replace_multiple_every_function_call_in(v, functions)))
                .collect(),
        ),

        MLIR::MakeFunction {
            name: fn_name,
            params: fn_params,
            body: fn_body,
        } => MLIR::MakeFunction {
            name: fn_name,
            params: fn_params,
            body: Box::new(replace_multiple_every_function_call_in(*fn_body, functions)),
        },

        MLIR::Let {
            name: let_name,
            value: let_value,
        } => MLIR::Let {
            name: let_name,
            value: Box::new(replace_multiple_every_function_call_in(
                *let_value, functions,
            )),
        },

        _ => mlir,
    }
}

/// Replace recursively each call to `name(args)` by the inlined body.
pub fn replace_every_function_call_in<'a>(
    mlir: MLIR<'a>,
    name: &'a str,
    new_value: &MLIR<'a>,
    new_params: &[&'a str],
) -> MLIR<'a> {
    match mlir {
        MLIR::FunctionCall {
            name: func_name,
            ref args,
        } if func_name == name => {
            let bound: Vec<(&'a str, MLIR<'a>)> = new_params
                .iter()
                .copied()
                .zip(args.iter().map(|arg| {
                    replace_every_function_call_in(arg.clone(), name, new_value, new_params)
                }))
                .collect();

            let result = bound.iter().fold(new_value.clone(), |acc, (k, v)| {
                replace_variable_with_value(&acc, k, v)
            });
            replace_every_function_call_in(result, name, new_value, new_params)
        }

        MLIR::FunctionCall { .. } => mlir,

        MLIR::MakeFunction {
            name: fn_name,
            params,
            body,
        } => MLIR::MakeFunction {
            name: fn_name,
            params,
            body: Box::new(replace_every_function_call_in(
                *body, name, new_value, new_params,
            )),
        },

        MLIR::Let {
            name: let_name,
            value,
        } => MLIR::Let {
            name: let_name,
            value: Box::new(replace_every_function_call_in(
                *value, name, new_value, new_params,
            )),
        },

        MLIR::Object(pairs) => MLIR::Object(
            pairs
                .into_iter()
                .map(|(k, v)| {
                    (
                        k,
                        replace_every_function_call_in(v, name, new_value, new_params),
                    )
                })
                .collect(),
        ),

        MLIR::Array(elems) => MLIR::Array(
            elems
                .into_iter()
                .map(|e| replace_every_function_call_in(e, name, new_value, new_params))
                .collect(),
        ),

        MLIR::Add { left, right } => MLIR::Add {
            left: Box::new(replace_every_function_call_in(
                *left, name, new_value, new_params,
            )),
            right: Box::new(replace_every_function_call_in(
                *right, name, new_value, new_params,
            )),
        },

        _ => mlir,
    }
}
