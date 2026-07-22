use std::collections::HashMap;

use jsonc_mlir::MLIR;

#[derive(Clone, Debug)]
/// This struct optimizes JSONC objects by inlining function calls and replacing
/// variables with their values.
pub struct ObjectOptimizer<'a> {
    pub objects: Vec<MLIR<'a>>,
    pub new_functions: HashMap<&'a str, MLIR<'a>>,
    pub signature_counts: HashMap<String, usize>,
    pub threshold: usize,
}

/// Replaces multiple variables with their corresponding values in the given MLIR expression.
///
/// # Arguments
///
/// * `mlir` - The MLIR expression to replace variables in.
/// * `replacements` - A map of variable names to their corresponding values.
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

        MLIR::Object(hm) => {
            let mut new_hm = Vec::with_capacity(hm.len());
            for (k, v) in hm {
                new_hm.push((k, replace_multiple_variables_with_values(v, replacements)));
            }
            MLIR::Object(new_hm)
        }

        MLIR::Array(elements) => {
            let mut new_elements = Vec::with_capacity(elements.len());
            for v in elements {
                new_elements.push(replace_multiple_variables_with_values(v, replacements));
            }
            MLIR::Array(new_elements)
        }

        MLIR::Add { left, right } => MLIR::Add {
            left: Box::new(replace_multiple_variables_with_values(*left, replacements)),
            right: Box::new(replace_multiple_variables_with_values(*right, replacements)),
        },

        MLIR::FunctionCall {
            name: fn_name,
            args,
        } => {
            let mut new_args = Vec::with_capacity(args.len());
            for v in args {
                new_args.push(replace_multiple_variables_with_values(v, replacements));
            }
            MLIR::FunctionCall {
                name: fn_name,
                args: new_args,
            }
        }

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
///
/// # Arguments
///
/// * `mlir` - The MLIR expression to replace variables in.
/// * `name` - The name of the variable to replace.
/// * `val` - The value to replace the variable with.
pub fn replace_variable_with_value<'a>(mlir: &MLIR<'a>, name: &'a str, val: &MLIR<'a>) -> MLIR<'a> {
    match mlir {
        MLIR::Variable(n) if *n == name => val.clone(),

        MLIR::Object(hm) => {
            let mut new_hm = Vec::with_capacity(hm.len());
            for (k, v) in hm {
                new_hm.push((*k, replace_variable_with_value(v, name, val)));
            }
            MLIR::Object(new_hm)
        }

        MLIR::Array(elements) => {
            let mut new_elements = Vec::with_capacity(elements.len());
            for v in elements {
                new_elements.push(replace_variable_with_value(v, name, val));
            }
            MLIR::Array(new_elements)
        }

        MLIR::Add { left, right } => MLIR::Add {
            left: Box::new(replace_variable_with_value(left, name, val)),
            right: Box::new(replace_variable_with_value(right, name, val)),
        },

        MLIR::FunctionCall {
            name: fn_name,
            args,
        } => {
            let mut new_args = Vec::with_capacity(args.len());
            for v in args {
                new_args.push(replace_variable_with_value(v, name, val));
            }
            MLIR::FunctionCall {
                name: *fn_name,
                args: new_args,
            }
        }

        MLIR::MakeFunction {
            name: fn_name,
            params,
            body,
        } => {
            let new_body = Box::new(replace_variable_with_value(body, name, val));
            MLIR::MakeFunction {
                name: *fn_name,
                params: params.clone(),
                body: new_body,
            }
        }

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
            signature_counts: HashMap::new(),
            threshold: 10,
        }
    }

    pub fn new_with_threshold(threshold: usize) -> Self {
        Self {
            objects: Vec::new(),
            new_functions: HashMap::new(),
            signature_counts: HashMap::new(),
            threshold,
        }
    }

    pub fn build_frequencies(&mut self, mlir: &MLIR<'a>) {
        let mut counts = HashMap::new();
        self.count_frequencies(mlir, true, &mut counts);
        self.signature_counts = counts;
    }

    /// Count the frequency of each node signature in the given MLIR expression.
    ///
    /// # Arguments
    ///
    /// * `mlir` - The MLIR expression to count node signatures in.
    /// * `root` - Whether the expression is the root of the object.
    /// * `counts` - A map to store the frequency of each node signature.
    fn count_frequencies(&self, mlir: &MLIR<'a>, root: bool, counts: &mut HashMap<String, usize>) {
        match mlir {
            MLIR::Object(hm) => {
                // Root objects are treated as separate entities because they might appear one time in the input,
                // while non-root objects are treated as part of the same entity.
                if root {
                    for (_, v) in hm {
                        self.count_frequencies(v, false, counts);
                    }
                    return;
                }

                let mut signature = String::new();
                for (k, v) in hm {
                    self.count_frequencies(v, false, counts);

                    let is_const = match v {
                        MLIR::Null => true,
                        MLIR::Bool(_) => true,
                        MLIR::Number(_) => true,
                        MLIR::Variable(n) if n.starts_with('s') => true,
                        MLIR::Array(el) if el.is_empty() => true,
                        _ => false,
                    };

                    if is_const {
                        let const_repr = match v {
                            MLIR::Null => "null".to_string(),
                            MLIR::Bool(b) => b.to_string(),
                            MLIR::Number(n) => n.to_bits().to_string(),
                            MLIR::Variable(n) => format!("var_{}", n),
                            MLIR::Array(_) => "empty_arr".to_string(),
                            _ => unreachable!(),
                        };
                        signature.push_str(&format!("_{}_{}", k, const_repr));
                    } else {
                        signature.push_str(&format!("_{}_var", k));
                    }
                }

                let safe_sig = signature
                    .replace('.', "_")
                    .replace('-', "_")
                    .replace(':', "_")
                    .replace('?', "_")
                    .replace('&', "_")
                    .replace('=', "_")
                    .replace('/', "_");
                let function_str = format!("fn{}", safe_sig);
                *counts.entry(function_str).or_insert(0) += 1;
            }
            MLIR::Array(vec) => {
                for v in vec {
                    self.count_frequencies(v, false, counts);
                }
            }
            MLIR::Add { left, right } => {
                self.count_frequencies(left, false, counts);
                self.count_frequencies(right, false, counts);
            }
            MLIR::FunctionCall { args, .. } => {
                for a in args {
                    self.count_frequencies(a, false, counts);
                }
            }
            MLIR::Let { value, .. } => {
                self.count_frequencies(value, false, counts);
            }
            _ => {}
        }
    }

    pub fn add_json(&mut self, mlir: MLIR<'a>) {
        self.objects.push(mlir);
    }

    /// Find objects that are similar to the given MLIR expression.
    ///
    /// # Arguments
    ///
    /// * `mlir` - The MLIR expression to find similar objects for.
    ///
    /// # Returns
    ///
    /// A vector of names of similar objects keys found.
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

    /// Format the output by collecting all new functions and the input MLIR expression.
    ///
    /// # Arguments
    ///
    /// * `mlir` - The MLIR expression to format.
    pub fn format_output(&self, mlir: MLIR<'a>) -> Vec<MLIR<'a>> {
        let mut result = self.new_functions.values().cloned().collect::<Vec<_>>();
        result.push(mlir);
        result
    }

    /// Optimize the given MLIR expression.
    ///
    /// # Arguments
    ///
    /// * `mlir` - The MLIR expression to optimize.
    /// * `root` - Whether the expression is the root of the object.
    pub fn optimize(&mut self, mlir: MLIR<'a>, root: bool) -> MLIR<'a> {
        match mlir {
            MLIR::Object(hm) => {
                if root {
                    let mut pairs = Vec::with_capacity(hm.len());
                    for (k, v) in hm {
                        pairs.push((k, self.optimize(v, false)));
                    }
                    return MLIR::Object(pairs);
                }

                let mut signature = String::new();
                let mut params = Vec::new();
                let mut args = Vec::new();
                let mut body_pairs = Vec::with_capacity(hm.len());
                let mut raw_pairs = Vec::with_capacity(hm.len());

                for (k, v) in hm {
                    let optimized_val = self.optimize(v, false);
                    raw_pairs.push((k, optimized_val.clone()));

                    let is_const = match &optimized_val {
                        MLIR::Null => true,
                        MLIR::Bool(_) => true,
                        MLIR::Number(_) => true,
                        MLIR::Variable(n) if n.starts_with('s') => true,
                        MLIR::Array(el) if el.is_empty() => true,
                        _ => false,
                    };

                    if is_const {
                        let const_repr = match &optimized_val {
                            MLIR::Null => "null".to_string(),
                            MLIR::Bool(b) => b.to_string(),
                            MLIR::Number(n) => n.to_bits().to_string(),
                            MLIR::Variable(n) => format!("var_{}", n),
                            MLIR::Array(_) => "empty_arr".to_string(),
                            _ => unreachable!(),
                        };
                        signature.push_str(&format!("_{}_{}", k, const_repr));
                        body_pairs.push((k, optimized_val));
                    } else {
                        signature.push_str(&format!("_{}_var", k));
                        params.push(k);
                        args.push(optimized_val);
                        body_pairs.push((k, MLIR::Variable(k)));
                    }
                }

                let safe_sig = signature
                    .replace('.', "_")
                    .replace('-', "_")
                    .replace(':', "_")
                    .replace('?', "_")
                    .replace('&', "_")
                    .replace('=', "_")
                    .replace('/', "_");
                let function_str: &'static str =
                    Box::leak(format!("fn{}", safe_sig).into_boxed_str());

                let count = *self.signature_counts.get(function_str).unwrap_or(&0);
                if count >= self.threshold {
                    if !self.new_functions.contains_key(function_str) {
                        self.new_functions.insert(
                            function_str,
                            MLIR::MakeFunction {
                                name: function_str,
                                params,
                                body: Box::new(MLIR::Object(body_pairs)),
                            },
                        );
                    }

                    MLIR::FunctionCall {
                        name: function_str,
                        args,
                    }
                } else {
                    MLIR::Object(raw_pairs)
                }
            }

            MLIR::Array(vec) => {
                let mut new_vec = Vec::with_capacity(vec.len());
                for v in vec {
                    new_vec.push(self.optimize(v, false));
                }
                MLIR::Array(new_vec)
            }

            MLIR::Add { left, right } => MLIR::Add {
                left: Box::new(self.optimize(*left, false)),
                right: Box::new(self.optimize(*right, false)),
            },

            MLIR::MakeFunction { name, params, body } => MLIR::MakeFunction { name, params, body },

            MLIR::FunctionCall { name, args } => {
                let mut new_args = Vec::with_capacity(args.len());
                for a in args {
                    new_args.push(self.optimize(a, false));
                }
                MLIR::FunctionCall {
                    name,
                    args: new_args,
                }
            }

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
