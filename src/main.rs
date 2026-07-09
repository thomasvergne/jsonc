use std::{collections::HashMap, io::Read, path::Path};

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

    let ratios = json_sizes
        .iter()
        .map(|(f, s)| f.clone() as f64 / s.clone() as f64)
        .collect::<Vec<f64>>();

    let total = ratios.iter().fold(0.0, |acc, r| acc + r);

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
        json_size as f64 / compressed_size as f64,
    );
}

fn perform_encoding(file_name: &str, output_path: &str) -> (usize, usize) {
    let data = Box::leak(std::fs::read_to_string(file_name).unwrap().into_boxed_str());

    let v = serde_json::from_str::<Value>(data).unwrap();

    let mlir = jsonc_mlir::from_json(&v);

    let mut str_optimizer = jsonc_string_opt::StringOptimizer::new();
    str_optimizer.traverse_and_collect_strings(&mlir);
    let optimized_mlir = str_optimizer.optimize(&mlir);

    let mut obj_optimizer = jsonc_object_opt::ObjectOptimizer::new();
    let optimized_mlir = obj_optimizer.optimize(optimized_mlir, IS_ROOT);
    let formatted_mlir = obj_optimizer.format_output(optimized_mlir.clone());

    let mut value_optimizer = jsonc_value_opt::ValueOptimizer::new();
    let optimized_mlir = formatted_mlir
        .iter()
        .map(|mlir| value_optimizer.optimize_all(&mlir))
        .collect::<Vec<_>>();
    let lets = value_optimizer.create_lets();

    let optimized_mlir = lets.into_iter().chain(optimized_mlir).collect::<Vec<_>>();
    let optimized_mlir = value_optimizer.remove_unused_variables(optimized_mlir);

    let mut compiler = jsonc_compiler::Compiler::new();
    let bytecode = compiler.compile_all(optimized_mlir.clone());

    let _ = jsonc_encoder::write_instrs(&bytecode, &compiler.value_pool, output_path);

    // Compare size of json data and jsonb data
    let json_size = data.len();
    let jsonb_size = std::fs::metadata(output_path).unwrap().len() as usize;

    display_compression_ratio(json_size, jsonb_size);

    return (json_size, jsonb_size);
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
        );

    let matches = cmd.get_matches();
    let matches = match matches.subcommand() {
        Some(("encode", matches)) => ("encode", matches),
        Some(("decode", matches)) => ("decode", matches),
        Some(("benchmark", matches)) => ("benchmark", matches),

        _ => unreachable!("clap should ensure we don't get here"),
    };

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
