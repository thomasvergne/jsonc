use std::collections::HashMap;

use jsonc_mlir::MLIR;
use jsonc_object_opt::{IS_ROOT, replace_variable_with_value};

pub struct ValueOptimizer<'a> {
    pub value_pool: Vec<MLIR<'a>>,
    pub value_counter: HashMap<usize, usize>,
    pub value_to_drop: Vec<usize>,
    /// Cache de noms de variables "v0", "v1", evite un Box::leak par appel.
    var_name_cache: Vec<&'static str>,
}

impl<'a> ValueOptimizer<'a> {
    pub fn new() -> Self {
        Self {
            value_pool: Vec::new(),
            value_counter: HashMap::new(),
            value_to_drop: Vec::new(),
            var_name_cache: Vec::new(),
        }
    }

    /// Retourne et met en cache la chaine statique "v{index}".
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
        // Collecte d'abord les (index, valeur) pour eviter le conflit d'emprunt
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

    pub fn optimize_all(&mut self, mlir: &MLIR<'a>) -> MLIR<'a> {
        let mut optimized = self.optimize(mlir, IS_ROOT);

        let mut lets = self.value_pool.clone();
        let pool_len = self.value_pool.len();

        for index in 0..pool_len {
            let count = *self.value_counter.get(&index).unwrap_or(&0);

            if count <= 1 {
                // Clone une seule fois depuis le pool original (non modifie)
                match self.value_pool[index].clone() {
                    MLIR::MakeFunction { name, body, params } => {
                        let body_owned = *body;
                        // Mutation en place : evite l'allocation d'un nouveau Vec par iteration
                        for v in lets.iter_mut() {
                            let old = std::mem::replace(v, MLIR::Null);
                            *v = replace_every_function_call_in(old, name, &body_owned, &params);
                        }
                        self.value_to_drop.push(index);
                    }

                    let_value => {
                        let var_str = self.var_name(index);
                        // Mutation en place ; val passe par reference, clone seulement si substitution
                        for v in lets.iter_mut() {
                            let old = std::mem::replace(v, MLIR::Null);
                            *v = replace_variable_with_value(&old, var_str, &let_value);
                        }
                    }
                }
            }
        }

        self.value_pool = lets.clone();

        for (index, let_) in lets.iter().enumerate() {
            if matches!(self.value_counter.get(&index), Some(c) if *c > 1) {
                continue;
            }

            if let MLIR::MakeFunction { name, params, body } = let_ {
                self.value_to_drop.push(index);

                optimized = replace_every_function_call_in(optimized, name, body, &params);

                self.value_counter.remove(&index);

                continue;
            }

            self.value_to_drop.push(index);
            let var_str = self.var_name(index);
            // let_ est une reference : pas de clone sauf si une substitution a lieu
            optimized = replace_variable_with_value(&optimized, var_str, let_);
            self.value_counter.remove(&index);
        }

        optimized
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
                if let Some(index) = self.value_pool.iter().position(|v| *v == *mlir) {
                    *self.value_counter.entry(index).or_insert(0) += 1;
                    self.create_variable(index)
                } else {
                    let index = self.value_pool.len();
                    self.value_pool.push(mlir.clone());
                    self.value_counter.insert(index, 1);
                    self.create_variable(index)
                }
            }

            MLIR::Array(elements) if !root => {
                if let Some(index) = self.value_pool.iter().position(|v| *mlir == *v) {
                    *self.value_counter.entry(index).or_insert(0) += 1;
                    self.create_variable(index)
                } else {
                    let unique_elements =
                        elements.iter().map(|e| self.optimize(e, false)).collect();
                    let index = self.value_pool.len();
                    self.value_pool.push(MLIR::Array(unique_elements));
                    self.value_counter.insert(index, 1);
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
                if let Some(index) = self.value_pool.iter().position(|v| *v == function) {
                    *self.value_counter.entry(index).or_insert(0) += 1;
                } else {
                    let index = self.value_pool.len();
                    self.value_counter.insert(index, 1);
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

            _ => mlir.clone(),
        }
    }
}

/// Remplace recursivement chaque appel a `name(args)` par le corps inline.
/// `new_value` et `new_params` sont passes par reference : on ne clone
/// `new_value` qu'au moment d'une substitution effective.
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
            // Associe chaque parametre formel a son argument (recursivement substitue)
            let bound: Vec<(&'a str, MLIR<'a>)> = new_params
                .iter()
                .copied()
                .zip(args.iter().map(|arg| {
                    replace_every_function_call_in(arg.clone(), name, new_value, new_params)
                }))
                .collect();

            // Substitue les parametres dans une copie fraiche de new_value
            let result = bound.iter().fold(new_value.clone(), |acc, (k, v)| {
                replace_variable_with_value(&acc, k, v)
            });

            // Recurse au cas ou le resultat contient encore des appels a `name`
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
