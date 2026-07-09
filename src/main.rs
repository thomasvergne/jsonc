use std::{collections::HashMap, io::Read, path::Path};

use jsonc_mlir::debug;
use jsonc_object_opt::IS_ROOT;
use serde_json::Value;

fn benchmark(folder: &str) {
    let files = std::fs::read_dir(folder).unwrap();

    let mut json_sizes = Vec::new();

    for file in files {
        let file = file.unwrap();
        let file_name = file.path();

        if file_name.extension() != Some(std::ffi::OsStr::new("json")) {
            continue;
        }

        let output = file_name.with_extension("jsonb");
        let (json_size, jsonb_size) =
            perform_encoding(&file_name.to_str().unwrap(), output.to_str().unwrap());

        json_sizes.push((json_size, jsonb_size));
    }

    if json_sizes.is_empty() {
        println!("Average compression ratio: 0.00");
        return;
    }

    let ratios = json_sizes
        .iter()
        .map(|(f, s)| *f as f64 / *s as f64)
        .collect::<Vec<f64>>();

    let total: f64 = ratios.iter().sum();

    let average = total as f64 / ratios.len() as f64;

    println!("Average compression ratio: {:.2}", average);
}

fn display_compression_ratio(json_size: usize, compressed_size: usize) {
    fn display_correct_value(value: usize) -> (usize, &'static str) {
        if value < 1024 {
            (value, "B")
        } else if value < 10 * 1024 * 1024 {
            (value / 1024, "KB")
        } else {
            (value / (1024 * 1024), "MB")
        }
    }

    let (correct_value, correct_unit) = display_correct_value(compressed_size);
    let (json_value, json_unit) = display_correct_value(json_size);

    println!(
        "Compression ratio: {} ({}) versus {} ({}). Ratio {:.2}",
        correct_value,
        correct_unit,
        json_value,
        json_unit,
        if compressed_size == 0 {
            0.0
        } else {
            json_size as f64 / compressed_size as f64
        },
    );
}

fn perform_encoding(file_name: &str, output_path: &str) -> (usize, usize) {
    let data = std::fs::read_to_string(file_name).unwrap();

    let v = serde_json::from_str::<Value>(&data);

    if let Err(e) = v {
        panic!("Failed to parse JSON: {}", e);
    }

    let v = v.unwrap();

    debug!("[1] Converting JSON to MLIR");
    let mlir = jsonc_mlir::from_json(&v);

    debug!("[2] Optimizing strings");
    let mut str_optimizer = jsonc_string_opt::StringOptimizer::new();
    str_optimizer.traverse_and_collect_strings(&mlir);
    let optimized_mlir = str_optimizer.optimize(&mlir);

    debug!("[3] Optimizing objects");
    let mut obj_optimizer = jsonc_object_opt::ObjectOptimizer::new();
    let optimized_mlir = obj_optimizer.optimize(optimized_mlir, IS_ROOT);
    let formatted_mlir = obj_optimizer.format_output(optimized_mlir);

    debug!("[4] Optimizing values");
    let mut value_optimizer = jsonc_value_opt::ValueOptimizer::new();
    let collected_mlir = value_optimizer.optimize_all(&formatted_mlir);
    let optimized_mlir = value_optimizer.optimize_program(collected_mlir);

    debug!("[5] Creating lets");
    let lets = value_optimizer.create_lets();

    let optimized_mlir = lets.into_iter().chain(optimized_mlir).collect::<Vec<_>>();

    debug!("[6] Removing unused variables");
    let optimized_mlir = value_optimizer.remove_unused_variables(optimized_mlir);

    let mut compiler = jsonc_compiler::Compiler::new();
    debug!("[7] Compiling to bytecode");
    let bytecode = compiler.compile_all(optimized_mlir);

    debug!("[8] Writing bytecode to file");
    let _ = jsonc_encoder::write_instrs(&bytecode, &compiler.value_pool, output_path);

    // Compare size of json data and jsonb data
    let json_size = data.len();
    let jsonb_size = std::fs::metadata(output_path).unwrap().len() as usize;

    display_compression_ratio(json_size, jsonb_size);

    (json_size, jsonb_size)
}

fn perform_decoding(file_name: &str, output_path: &str) {
    let mut decoder = jsonc_decoder::Decoder::new(vec![], vec![]);
    let mut file = std::fs::File::open(file_name).unwrap();
    let mut buf = Vec::new();
    let _ = file.read_to_end(&mut buf);

    match decoder.decode(buf) {
        Ok(_) => match decoder.to_mlir() {
            Ok(mlir) => {
                let result = jsonc_mlir::multiple_to_json(&mlir, &mut HashMap::new());
                let result = serde_json::to_string(&result).unwrap();

                std::fs::write(output_path, result).unwrap();
            }
            Err(e) => println!("Error: {:?}", e),
        },
        Err(e) => println!("Error: {:?}", e),
    }
}

fn minify(input_path: &str, output_path: &str) {
    let mut file = std::fs::File::open(input_path).unwrap();
    let mut buf = Vec::new();
    let _ = file.read_to_end(&mut buf);

    let json = serde_json::from_slice::<Value>(&buf).unwrap();
    let result = serde_json::to_string(&json).unwrap();

    std::fs::write(output_path, result).unwrap();
}

fn compare_json_files(path1: &str, path2: &str) {
    let data1 = std::fs::read_to_string(path1).unwrap();
    let data2 = std::fs::read_to_string(path2).unwrap();
    let v1: serde_json::Value = serde_json::from_str(&data1).unwrap();
    let v2: serde_json::Value = serde_json::from_str(&data2).unwrap();
    if v1 == v2 {
        println!("The JSON files are semantically identical!");
    } else {
        println!("The JSON files are DIFFERENT!");
        diff_values(&v1, &v2, "");
    }
}

fn diff_values(v1: &serde_json::Value, v2: &serde_json::Value, path: &str) {
    match (v1, v2) {
        (serde_json::Value::Object(m1), serde_json::Value::Object(m2)) => {
            for (k, val1) in m1 {
                let next_path = format!("{}.{}", path, k);
                if let Some(val2) = m2.get(k) {
                    diff_values(val1, val2, &next_path);
                } else {
                    println!("Key {} is missing in second JSON", next_path);
                }
            }
            for k in m2.keys() {
                if !m1.contains_key(k) {
                    println!("Key {}.{} is missing in first JSON", path, k);
                }
            }
        }
        (serde_json::Value::Array(a1), serde_json::Value::Array(a2)) => {
            if a1.len() != a2.len() {
                println!(
                    "Array at {} has different length: {} vs {}",
                    path,
                    a1.len(),
                    a2.len()
                );
            }
            let min_len = std::cmp::min(a1.len(), a2.len());
            for i in 0..min_len {
                diff_values(&a1[i], &a2[i], &format!("{}[{}]", path, i));
            }
        }
        (val1, val2) => {
            if val1 != val2 {
                println!("Difference at {}: {:?} vs {:?}", path, val1, val2);
            }
        }
    }
}

fn main() {
    let cmd = clap::Command::new("cargo")
        .bin_name("jsonc")
        .styles(CLAP_STYLING)
        .subcommand_required(true)
        .subcommand(
            clap::command!("encode")
                .arg(
                    clap::Arg::new("input")
                        .required(true)
                        .index(1)
                        .value_parser(clap::value_parser!(std::path::PathBuf)),
                )
                .arg(
                    clap::arg!(--"output" <PATH>)
                        .value_parser(clap::value_parser!(std::path::PathBuf)),
                ),
        )
        .subcommand(
            clap::command!("decode")
                .arg(
                    clap::Arg::new("input")
                        .required(true)
                        .index(1)
                        .value_parser(clap::value_parser!(std::path::PathBuf)),
                )
                .arg(
                    clap::arg!(--"output" <PATH>)
                        .value_parser(clap::value_parser!(std::path::PathBuf)),
                ),
        )
        .subcommand(
            clap::command!("benchmark").arg(
                clap::Arg::new("input")
                    .required(true)
                    .index(1)
                    .value_parser(clap::value_parser!(std::path::PathBuf)),
            ),
        )
        .subcommand(
            clap::command!("minify").arg(
                clap::Arg::new("input")
                    .required(true)
                    .index(1)
                    .value_parser(clap::value_parser!(std::path::PathBuf)),
            ),
        )
        .subcommand(
            clap::command!("compare")
                .arg(
                    clap::Arg::new("input1")
                        .required(true)
                        .index(1)
                        .value_parser(clap::value_parser!(std::path::PathBuf)),
                )
                .arg(
                    clap::Arg::new("input2")
                        .required(true)
                        .index(2)
                        .value_parser(clap::value_parser!(std::path::PathBuf)),
                ),
        );

    let matches = cmd.get_matches();
    let matches = match matches.subcommand() {
        Some(("encode", matches)) => ("encode", matches),
        Some(("decode", matches)) => ("decode", matches),
        Some(("benchmark", matches)) => ("benchmark", matches),
        Some(("minify", matches)) => ("minify", matches),
        Some(("compare", matches)) => ("compare", matches),

        _ => unreachable!("clap should ensure we don't get here"),
    };

    if matches.0 == "compare" {
        let input1 = matches.1.get_one::<std::path::PathBuf>("input1").unwrap();
        let input2 = matches.1.get_one::<std::path::PathBuf>("input2").unwrap();
        compare_json_files(input1.to_str().unwrap(), input2.to_str().unwrap());
        return;
    }

    let input_path = matches.1.get_one::<std::path::PathBuf>("input");
    let output_path = matches
        .1
        .get_one::<std::path::PathBuf>("output")
        .map(|v| v.clone())
        .unwrap_or_else(|| {
            let path = Path::new(input_path.unwrap());
            path.with_extension(if matches.0 == "encode" {
                "jsonb"
            } else {
                "json"
            })
        });

    match matches.0 {
        "encode" => {
            perform_encoding(
                input_path.unwrap().to_str().unwrap(),
                output_path.to_str().unwrap(),
            );
        }
        "decode" => perform_decoding(
            input_path.unwrap().to_str().unwrap(),
            output_path.to_str().unwrap(),
        ),
        "benchmark" => {
            benchmark(input_path.unwrap().to_str().unwrap());
        }
        "minify" => {
            minify(
                input_path.unwrap().to_str().unwrap(),
                output_path.to_str().unwrap(),
            );
        }
        _ => unreachable!("clap should ensure we don't get here"),
    }
}

// See also `clap_cargo::style::CLAP_STYLING`
pub const CLAP_STYLING: clap::builder::styling::Styles = clap::builder::styling::Styles::styled()
    .header(clap_cargo::style::HEADER)
    .usage(clap_cargo::style::USAGE)
    .literal(clap_cargo::style::LITERAL)
    .placeholder(clap_cargo::style::PLACEHOLDER)
    .error(clap_cargo::style::ERROR)
    .valid(clap_cargo::style::VALID)
    .invalid(clap_cargo::style::INVALID);
