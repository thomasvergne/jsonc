use std::{collections::HashMap, fmt::Debug};

use serde_json::{Map, json};

#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        println!($($arg)*);
    };
}

#[derive(Clone, PartialEq)]
pub enum MLIR<'a> {
    String(&'a str),
    Array(Vec<MLIR<'a>>),
    Object(Vec<(&'a str, MLIR<'a>)>),
    Null,
    Number(f64),
    Bool(bool),

    // MLIR related nodes
    Variable(&'a str),
    FunctionCall {
        name: &'a str,
        args: Vec<MLIR<'a>>,
    },
    MakeFunction {
        name: &'a str,
        params: Vec<&'a str>,
        body: Box<MLIR<'a>>,
    },

    Let {
        name: &'a str,
        value: Box<MLIR<'a>>,
    },

    Add {
        left: Box<MLIR<'a>>,
        right: Box<MLIR<'a>>,
    },
}

impl<'a> Debug for MLIR<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MLIR::Add { left, right } => {
                write!(f, "{left:?} + {right:?}")
            }

            MLIR::Bool(b) => write!(f, "{}", b),
            MLIR::Number(n) => write!(f, "{}", n),
            MLIR::String(s) => write!(f, "{:?}", s),
            MLIR::Null => write!(f, "null"),

            MLIR::Variable(name) => write!(f, "{}", name),
            MLIR::FunctionCall { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", arg)?;
                }
                write!(f, ")")
            }
            MLIR::MakeFunction { name, params, body } => {
                write!(f, "function {}(", name)?;

                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, ")")?;
                write!(f, "{{ {:?} }}", body)
            }
            MLIR::Let { name, value } => write!(f, "const {} = {:?};", name, value),

            MLIR::Array(elements) => {
                write!(f, "[")?;
                for (i, element) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", element)?;
                }
                write!(f, "]")?;
                Ok(())
            }
            MLIR::Object(pairs) => {
                write!(f, "{{")?;
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {:?}", k, v)?;
                }
                write!(f, "}}")?;
                Ok(())
            }
        }
    }
}

/// Converting a JSON value to MLIR.
pub fn from_json<'a>(value: &'a serde_json::Value) -> MLIR<'a> {
    match value {
        serde_json::Value::String(s) => MLIR::String(s),
        serde_json::Value::Array(arr) => {
            let elements = arr.iter().map(|v| from_json(v)).collect();
            MLIR::Array(elements)
        }
        serde_json::Value::Object(obj) => {
            let mut pairs = Vec::new();
            for (k, v) in obj {
                pairs.push((k.as_str(), from_json(v)));
            }
            MLIR::Object(pairs)
        }
        serde_json::Value::Null => MLIR::Null,
        serde_json::Value::Number(n) => MLIR::Number(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::Bool(b) => MLIR::Bool(*b),
    }
}

/// Environment for resolving MLIR variable references.
///
/// This enum represents the environment in which MLIR variable references are resolved.
/// It supports nested environments for scoping.
#[derive(Debug)]
pub enum Env<'a, 'b> {
    Root(&'b mut HashMap<&'a str, MLIR<'a>>),
    Nested {
        parent: &'b Env<'a, 'b>,
        bindings: HashMap<&'a str, MLIR<'a>>,
    },
}

impl<'a, 'b> Env<'a, 'b> {
    /// Returns the value of the given variable name, if it exists in this environment or its parent.
    pub fn get(&self, name: &str) -> Option<&MLIR<'a>> {
        match self {
            Env::Root(map) => map.get(name),
            Env::Nested { parent, bindings } => {
                if name.starts_with('l') {
                    bindings.get(name)
                } else {
                    bindings.get(name).or_else(|| parent.get(name))
                }
            }
        }
    }

    /// Inserts a new variable binding into this environment.
    ///
    /// # Panics
    ///
    /// Panics if the environment is a nested environment.
    pub fn insert(&mut self, name: &'a str, val: MLIR<'a>) {
        match self {
            Env::Root(map) => {
                map.insert(name, val);
            }
            Env::Nested { .. } => {
                panic!("Cannot insert into nested env");
            }
        }
    }
}

/// Converts an MLIR node to a JSON value.
pub fn to_json<'a, 'b>(mlir: &MLIR<'a>, env: &mut Env<'a, 'b>) -> serde_json::Value {
    match mlir {
        MLIR::Add { left, right } => {
            let l_val = to_json(left, env);
            let r_val = to_json(right, env);
            match (l_val, r_val) {
                (serde_json::Value::Number(l), serde_json::Value::Number(r)) => {
                    json!(l.as_f64().unwrap_or(0.0) + r.as_f64().unwrap_or(0.0))
                }
                (serde_json::Value::String(l), serde_json::Value::String(r)) => {
                    json!(l.to_string() + &r)
                }
                (serde_json::Value::Null, serde_json::Value::Null) => serde_json::Value::Null,
                (serde_json::Value::Null, _) => serde_json::Value::Null,
                (_, serde_json::Value::Null) => serde_json::Value::Null,
                _ => serde_json::Value::Null,
            }
        }

        MLIR::String(s) => json!(s),
        MLIR::Array(elements) => {
            json!(elements.iter().map(|e| to_json(e, env)).collect::<Vec<_>>())
        }
        MLIR::Object(obj) => {
            let mut new_obj = Map::new();

            for (k, v) in obj.iter() {
                new_obj.insert(k.to_string(), to_json(v, env));
            }

            serde_json::Value::Object(new_obj)
        }
        MLIR::Null => serde_json::Value::Null,
        MLIR::Number(n) => json!(n),
        MLIR::Bool(b) => json!(b),
        MLIR::Variable(name) => {
            let value = env.get(name).unwrap().clone();
            to_json(&value, env)
        }
        MLIR::FunctionCall { name, args } => {
            let Some(function) = env.get(name) else {
                eprintln!("function not found: {:?}", name);
                return serde_json::Value::Null;
            };

            if let MLIR::MakeFunction { params, body, .. } = function {
                let mut new_env = HashMap::new();
                for (k, v) in params.iter().zip(args.iter()) {
                    new_env.insert(*k, v.clone());
                }
                let mut env_wrapper = Env::Nested {
                    parent: env,
                    bindings: new_env,
                };
                let result = to_json(&body, &mut env_wrapper);
                return result;
            }

            return serde_json::Value::Null;
        }
        MLIR::MakeFunction { name, params, body } => {
            env.insert(
                name,
                MLIR::MakeFunction {
                    name,
                    params: params.clone(),
                    body: body.clone(),
                },
            );

            serde_json::Value::Null
        }
        MLIR::Let { name, value } => {
            env.insert(name, *value.clone());

            serde_json::Value::Null
        }
    }
}

/// Converts multiple MLIR nodes to a JSON value.
///
/// # Arguments
///
/// * `mlir` - The MLIR nodes to convert.
/// * `env` - The environment for resolving variable references.
pub fn multiple_to_json<'a>(
    mlir: &[MLIR<'a>],
    env: &mut HashMap<&'a str, MLIR<'a>>,
) -> serde_json::Value {
    let mut result = serde_json::Value::Null;
    let mut env_wrapper = Env::Root(env);
    for expr in mlir {
        result = to_json(expr, &mut env_wrapper);
    }

    result
}

/// Implements equality for MLIR nodes.
impl<'a> Eq for MLIR<'a> {}

/// Implements hashing for MLIR nodes.
impl<'a> std::hash::Hash for MLIR<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            MLIR::String(s) => {
                0u8.hash(state);
                s.hash(state);
            }
            MLIR::Array(arr) => {
                1u8.hash(state);
                arr.hash(state);
            }
            MLIR::Object(obj) => {
                2u8.hash(state);
                obj.hash(state);
            }
            MLIR::Null => {
                3u8.hash(state);
            }
            MLIR::Number(n) => {
                4u8.hash(state);
                n.to_bits().hash(state);
            }
            MLIR::Bool(b) => {
                5u8.hash(state);
                b.hash(state);
            }
            MLIR::Variable(s) => {
                6u8.hash(state);
                s.hash(state);
            }
            MLIR::FunctionCall { name, args } => {
                7u8.hash(state);
                name.hash(state);
                args.hash(state);
            }
            MLIR::MakeFunction { name, params, body } => {
                8u8.hash(state);
                name.hash(state);
                params.hash(state);
                body.hash(state);
            }
            MLIR::Let { name, value } => {
                9u8.hash(state);
                name.hash(state);
                value.hash(state);
            }
            MLIR::Add { left, right } => {
                10u8.hash(state);
                left.hash(state);
                right.hash(state);
            }
        }
    }
}
