use std::collections::{HashMap, HashSet};

use jsonc_mlir::MLIR;
use jsonc_object_opt::{IS_ROOT, replace_multiple_variables_with_values};

/// This structure holds the state of the value optimizer, including the value pool,
/// value counter, and value to drop sets.
/// It also holds a cache of variable names for reuse, and a mapping of values to their
/// indices in the value pool.
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

    /// Creates a [`MLIR::Let`] expression for each value in the value pool that is not a
    /// function definition and is not marked for dropping. Then, it performs a topological
    /// sort on the let expressions to determine the order in which they should be evaluated.
    /// Finally, it returns the sorted let expressions as a [`Vec`].
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

        // 1. Build map from variable name to its entry index and value
        let mut name_to_entry = HashMap::new();
        for (index, value) in &entries {
            name_to_entry.insert(self.var_name(*index), value.clone());
        }

        // 2. DFS topological sort
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();
        let mut sorted_names = Vec::new();

        fn visit<'a>(
            name: &'a str,
            name_to_entry: &HashMap<&'a str, MLIR<'a>>,
            visited: &mut HashSet<&'a str>,
            temp_visited: &mut HashSet<&'a str>,
            sorted_names: &mut Vec<&'a str>,
        ) {
            if visited.contains(name) {
                return;
            }
            if temp_visited.contains(name) {
                // Cycle detected, but should be impossible in hierarchical JSON
                return;
            }
            if let Some(value) = name_to_entry.get(name) {
                temp_visited.insert(name);
                let mut refs = HashSet::new();
                collect_referenced_variables(value, &mut refs);
                for dep in refs {
                    visit(dep, name_to_entry, visited, temp_visited, sorted_names);
                }
                temp_visited.remove(name);
                visited.insert(name);
                sorted_names.push(name);
            }
        }

        for name in name_to_entry.keys() {
            visit(
                *name,
                &name_to_entry,
                &mut visited,
                &mut temp_visited,
                &mut sorted_names,
            );
        }

        // 3. Construct Let nodes in sorted order
        let mut lets = Vec::new();
        for name in sorted_names {
            lets.push(MLIR::Let {
                name,
                value: Box::new(name_to_entry.get(name).unwrap().clone()),
            });
        }

        lets
    }

    /// Main entry point to optimize an MLIR expression.
    pub fn collect(&mut self, mlir: &MLIR<'a>) -> MLIR<'a> {
        self.optimize(mlir, IS_ROOT, false)
    }

    /// Optimizes a list of MLIR expressions.
    pub fn optimize_all(&mut self, mlirs: &Vec<MLIR<'a>>) -> Vec<MLIR<'a>> {
        let mut result = Vec::new();
        for mlir in mlirs.iter() {
            // debug!("Collecting values...");
            result.push(self.collect(mlir));
        }
        result
    }

    /// Resolves a [`MLIR::Let`] expression by recursively resolving its value and dependencies.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the [`MLIR::Let`] expression to resolve.
    /// * `lets` - The list of [`MLIR::Let`] expressions in the program.
    /// * `value_counter` - A [`HashMap`] that keeps track of the number of references to each value.
    /// * `resolved_replacements` - A [`HashMap`] that stores resolved variable replacements.
    /// * `resolved_functions` - A [`HashMap`] that stores resolved function definitions.
    /// * `resolved_lets` - A [`HashMap`] that stores resolved [`MLIR::Let`] expressions.
    /// * `resolving` - A [`HashSet`] that keeps track of the indices of [`MLIR::Let`] expressions currently being resolved.
    fn resolve_let(
        &mut self,
        index: usize,
        lets: &[MLIR<'a>],
        value_counter: &mut HashMap<usize, usize>,
        resolved_replacements: &mut HashMap<&'a str, MLIR<'a>>,
        resolved_functions: &mut HashMap<&'a str, (MLIR<'a>, Vec<&'a str>)>,
        resolved_lets: &mut HashMap<usize, MLIR<'a>>,
        resolving: &mut HashSet<usize>,
    ) {
        // Check if the let expression has already been resolved or is currently being resolved
        if resolved_lets.contains_key(&index) {
            return;
        }
        if resolving.contains(&index) {
            return;
        }
        resolving.insert(index);

        let old = lets[index].clone();

        // Find dependencies in 'old'
        let mut deps = HashSet::new();
        collect_referenced_variables(&old, &mut deps);

        // Resolve dependencies recursively
        for dep_name in deps {
            if dep_name.starts_with('v')
                && let Ok(dep_idx) = dep_name[1..].parse::<usize>()
                && dep_idx < lets.len()
            {
                self.resolve_let(
                    dep_idx,
                    lets,
                    value_counter,
                    resolved_replacements,
                    resolved_functions,
                    resolved_lets,
                    resolving,
                );
            }
        }

        // Inline let values and function calls that have been marked
        // as single-use.
        let val = replace_multiple_variables_with_values(old, resolved_replacements);
        let resolved_let = replace_multiple_every_function_call_in(val, resolved_functions);

        // If the let value is used only once, mark it for dropping and
        // inline it into the surrounding scope.
        let count = *value_counter.get(&index).unwrap_or(&0);
        if count <= 1 {
            self.value_to_drop.insert(index);
            match &resolved_let {
                MLIR::MakeFunction { name, params, body } => {
                    resolved_functions.insert(*name, (*body.clone(), params.clone()));
                }
                let_value => {
                    let var_name = self.var_name(index);
                    resolved_replacements.insert(var_name, let_value.clone());
                }
            }
            value_counter.remove(&index);
        }

        // Mark the let value as resolved and insert it into the resolved lets map.
        resolved_lets.insert(index, resolved_let);
        resolving.remove(&index);
    }

    /// Optimize the program by resolving let values and inlining function calls.
    /// Returns the optimized MLIR.
    ///
    /// # Arguments
    ///
    /// * `formatted_mlir` - The MLIR to optimize.
    pub fn optimize_program(&mut self, mut formatted_mlir: Vec<MLIR<'a>>) -> Vec<MLIR<'a>> {
        let lets = self.value_pool.clone();
        let pool_len = lets.len();

        // Defining prerequired variables to inline let values and function calls.
        let mut resolved_replacements = HashMap::new();
        let mut resolved_functions = HashMap::new();
        let mut resolved_lets = HashMap::new();
        let mut resolving = HashSet::new();
        let mut value_counter = self.value_counter.clone();

        // For each let value in the pool, resolve it and store the result in resolved_lets.
        for index in 0..pool_len {
            self.resolve_let(
                index,
                &lets,
                &mut value_counter,
                &mut resolved_replacements,
                &mut resolved_functions,
                &mut resolved_lets,
                &mut resolving,
            );
        }

        self.value_counter = value_counter;

        // Construct final self.value_pool using the resolved_lets in the correct index order
        let mut final_lets = Vec::with_capacity(pool_len);
        for index in 0..pool_len {
            final_lets.push(resolved_lets.remove(&index).unwrap_or(MLIR::Null));
        }

        // Inline function calls in the final self.value_pool.
        // Repeat this process up to 5 times to handle nested function calls.
        //
        // This is done this way to avoid infinite recursion, stack overflow,
        // excessive memory usage or other potential issues with deep recursion.
        for _ in 0..5 {
            let mut updated = HashMap::new();
            for (name, (body, params)) in resolved_functions.iter() {
                let resolved_body =
                    replace_multiple_every_function_call_in(body.clone(), &resolved_functions);
                updated.insert(*name, (resolved_body, params.clone()));
            }
            resolved_functions = updated;
        }

        // Replace let values and function calls in the final self.value_pool.
        for val in final_lets.iter_mut() {
            let old = std::mem::replace(val, MLIR::Null);
            *val = replace_multiple_every_function_call_in(old, &resolved_functions);
        }

        self.value_pool = final_lets;

        // Replace let values and function calls in the final expression.
        for mlir in formatted_mlir.iter_mut() {
            let old = std::mem::replace(mlir, MLIR::Null);
            let val = replace_multiple_variables_with_values(old, &resolved_replacements);
            *mlir = replace_multiple_every_function_call_in(val, &resolved_functions);
        }

        formatted_mlir
    }

    /// Dead-code elimination: this simply removes unused variables and function calls
    /// at top-level from the final expression, except the last expression that represents
    /// the result of the program.
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

    /// Optimizes the given MLIR by replacing let values and function calls,
    /// and removing unused variables at top-level.
    pub fn optimize(&mut self, mlir: &MLIR<'a>, root: bool, inside_function: bool) -> MLIR<'a> {
        match mlir {
            MLIR::Array(elements) if !root => {
                let mut unique_elements = Vec::with_capacity(elements.len());
                for e in elements {
                    unique_elements.push(self.optimize(e, false, inside_function));
                }
                let array_mlir = MLIR::Array(unique_elements);

                // This portion of code checks if the array has already been optimized,
                // and if so, returns a cached variable instead of optimizing again.
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
                let mut optimized_args = Vec::with_capacity(args.len());
                for a in args {
                    optimized_args.push(self.optimize(a, false, inside_function));
                }
                let function = MLIR::FunctionCall {
                    name: *name,
                    args: optimized_args,
                };

                // We check for root as a special case, as we don't want to optimize
                // function calls at the top-level (they are treated as regular expressions).
                if !root {
                    if let Some(&index) = self.value_indices.get(&function) {
                        *self.value_counter.entry(index).or_insert(0) += 1;
                        self.create_variable(index)
                    } else {
                        let index = self.value_pool.len();
                        self.value_counter.insert(index, 1);
                        self.value_indices.insert(function.clone(), index);
                        self.value_pool.push(function);
                        self.create_variable(index)
                    }
                } else {
                    function
                }
            }

            MLIR::MakeFunction { name, params, body } => {
                let optimized_body = Box::new(self.optimize(body, false, true));
                MLIR::MakeFunction {
                    name: *name,
                    params: params.clone(),
                    body: optimized_body,
                }
            }

            MLIR::Add { left, right } => MLIR::Add {
                left: Box::new(self.optimize(left, false, inside_function)),
                right: Box::new(self.optimize(right, false, inside_function)),
            },

            MLIR::Let { name, value } => {
                let optimized_value = self.optimize(value, false, inside_function);
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
                let mut optimized_obj = Vec::with_capacity(obj.len());
                for (k, v) in obj {
                    optimized_obj.push((*k, self.optimize(v, false, inside_function)));
                }
                let obj_mlir = MLIR::Object(optimized_obj);
                if !root && !inside_function {
                    if let Some(&index) = self.value_indices.get(&obj_mlir) {
                        *self.value_counter.entry(index).or_insert(0) += 1;
                        self.create_variable(index)
                    } else {
                        let index = self.value_pool.len();
                        self.value_counter.insert(index, 1);
                        self.value_indices.insert(obj_mlir.clone(), index);
                        self.value_pool.push(obj_mlir);
                        self.create_variable(index)
                    }
                } else {
                    obj_mlir
                }
            }

            MLIR::Array(elements) => {
                let mut optimized_elements = Vec::with_capacity(elements.len());
                for e in elements {
                    optimized_elements.push(self.optimize(e, false, inside_function));
                }
                MLIR::Array(optimized_elements)
            }

            _ => mlir.clone(),
        }
    }
}

/// Collects all referenced variables in the given MLIR expression.
///
/// # Arguments
///
/// * `mlir` - The MLIR expression to collect referenced variables from.
/// * `refs` - A set to store the collected referenced variables.
pub fn collect_referenced_variables<'a>(mlir: &MLIR<'a>, refs: &mut HashSet<&'a str>) {
    match mlir {
        MLIR::Variable(name) => {
            refs.insert(*name);
        }
        MLIR::Array(arr) => {
            for e in arr {
                collect_referenced_variables(e, refs);
            }
        }
        MLIR::Object(obj) => {
            for (_, v) in obj {
                collect_referenced_variables(v, refs);
            }
        }
        MLIR::Add { left, right } => {
            collect_referenced_variables(left, refs);
            collect_referenced_variables(right, refs);
        }
        MLIR::Let { value, .. } => {
            collect_referenced_variables(value, refs);
        }
        MLIR::MakeFunction { body, .. } => {
            collect_referenced_variables(body, refs);
        }
        MLIR::FunctionCall { args, .. } => {
            for arg in args {
                collect_referenced_variables(arg, refs);
            }
        }
        _ => {}
    }
}

/// Replaces multiple variables with their corresponding values in the given MLIR expression.
///
/// # Arguments
///
/// * `mlir` - The MLIR expression to replace variables in.
/// * `functions` - A map of function names to their corresponding MLIR expression and parameters.
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
                let mut bound = HashMap::with_capacity(new_params.len());
                for (p, a) in new_params.iter().zip(args.iter()) {
                    bound.insert(*p, a.clone());
                }
                let result = replace_multiple_variables_with_values(new_value.clone(), &bound);
                replace_multiple_every_function_call_in(result, functions)
            } else {
                let mut optimized_args = Vec::with_capacity(args.len());
                for e in args {
                    optimized_args.push(replace_multiple_every_function_call_in(e, functions));
                }
                MLIR::FunctionCall {
                    name: func_name,
                    args: optimized_args,
                }
            }
        }

        MLIR::Add { left, right } => MLIR::Add {
            left: Box::new(replace_multiple_every_function_call_in(*left, functions)),
            right: Box::new(replace_multiple_every_function_call_in(*right, functions)),
        },

        MLIR::Array(elements) => {
            let mut optimized_elements = Vec::with_capacity(elements.len());
            for e in elements {
                optimized_elements.push(replace_multiple_every_function_call_in(e, functions));
            }
            MLIR::Array(optimized_elements)
        }

        MLIR::Object(hm) => {
            let mut optimized_hm = Vec::with_capacity(hm.len());
            for (k, v) in hm {
                optimized_hm.push((k, replace_multiple_every_function_call_in(v, functions)));
            }
            MLIR::Object(optimized_hm)
        }

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
///
/// # Arguments
///
/// * `mlir` - The MLIR expression to replace function calls in.
/// * `name` - The name of the function to replace.
/// * `new_value` - The new value to replace the function call with.
/// * `new_params` - The new parameters to replace the function call with.
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
            let mut bound = HashMap::with_capacity(new_params.len());
            for (p, a) in new_params.iter().zip(args.iter()) {
                bound.insert(*p, a.clone());
            }

            let result = replace_multiple_variables_with_values(new_value.clone(), &bound);
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

        MLIR::Object(pairs) => {
            let mut optimized_pairs = Vec::with_capacity(pairs.len());
            for (k, v) in pairs {
                optimized_pairs.push((
                    k,
                    replace_every_function_call_in(v, name, new_value, new_params),
                ));
            }
            MLIR::Object(optimized_pairs)
        }

        MLIR::Array(elems) => {
            let mut optimized_elems = Vec::with_capacity(elems.len());
            for e in elems {
                optimized_elems.push(replace_every_function_call_in(
                    e, name, new_value, new_params,
                ));
            }
            MLIR::Array(optimized_elems)
        }

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
