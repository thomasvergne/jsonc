use std::{collections::HashMap, fmt::Debug};

use serde_json::json;

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

// Use owned MLIR values in the environment to avoid returning references to temporaries
pub fn to_json<'a>(mlir: &MLIR<'a>, env: &mut HashMap<&'a str, MLIR<'a>>) -> serde_json::Value {
    match mlir {
        MLIR::String(s) => serde_json::Value::String(s.to_string()),
        MLIR::Array(arr) => serde_json::Value::Array(arr.iter().map(|v| to_json(v, env)).collect()),
        MLIR::Object(obj) => {
            let mut map = serde_json::Map::new();
            for (k, v) in obj {
                map.insert(k.to_string(), to_json(v, env));
            }
            serde_json::Value::Object(map)
        }
        MLIR::Null => serde_json::Value::Null,
        MLIR::Number(n) => {
            if n.fract() == 0.0 && *n >= (i64::MIN as f64) && *n <= (i64::MAX as f64) {
                serde_json::Value::Number(serde_json::Number::from(*n as i64))
            } else {
                match serde_json::Number::from_f64(*n) {
                    Some(num) => serde_json::Value::Number(num),
                    None => serde_json::Value::Null,
                }
            }
        }
        MLIR::Bool(b) => serde_json::Value::Bool(*b),
        MLIR::Variable(name) => {
            // Clone uniquement la valeur trouvee, pas tout l'env
            match env.get(*name).cloned() {
                Some(val) => to_json(&val, env),
                None => serde_json::Value::Null,
            }
        }
        MLIR::FunctionCall { name, args } => match env.get(*name) {
            Some(MLIR::MakeFunction { body, params, .. }) => {
                // clone the environment and insert parameter bindings (owned MLIRs)
                let mut map = env.clone();
                for (param, arg) in params.iter().zip(args.iter()) {
                    map.insert(*param, arg.clone());
                }

                to_json(&body, &mut map)
            }
            _ => serde_json::Value::Null,
        },
        MLIR::MakeFunction { params, name, body } => {
            env.insert(
                name,
                MLIR::MakeFunction {
                    params: params.clone(),
                    name: *name,
                    body: body.clone(),
                },
            );

            serde_json::Value::Null
        }
        MLIR::Let { name, value } => {
            env.insert(name, *value.clone());
            serde_json::Value::Null
        }
        MLIR::Add { left, right } => match (to_json(left, env), to_json(right, env)) {
            (serde_json::Value::Number(l), serde_json::Value::Number(r)) => {
                json!(l.as_f64().unwrap_or(0.0) + r.as_f64().unwrap_or(0.0))
            }
            (serde_json::Value::Null, serde_json::Value::Null) => serde_json::Value::Null,
            (serde_json::Value::Null, _) => serde_json::Value::Null,
            (_, serde_json::Value::Null) => serde_json::Value::Null,
            (serde_json::Value::String(s1), serde_json::Value::String(s2)) => {
                json!(s1.to_string() + &s2)
            }
            _ => serde_json::Value::Null,
        },
    }
}

pub fn multiple_to_json<'a>(
    mlir: &[MLIR<'a>],
    env: &mut HashMap<&'a str, MLIR<'a>>,
) -> serde_json::Value {
    let mut result = serde_json::Value::Null;
    for expr in mlir {
        result = to_json(expr, env);
    }

    result
}

impl<'a> Eq for MLIR<'a> {}

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
