# JSON Compiler

A minimalist bytecode encoding JSON format designed to optimize JSON files, currently mostly based on motif analysis. It is delivered with a CLI that offers JSON files encoding and JSONB files decoding.

## Features

- JSON file encoding
- JSONB file decoding
- Benchmarking tool 

## Run Locally

Clone the project

```bash
  git clone https://github.com/thomasvergne/jsonc
```

Go to the project directory

```bash
  cd jsonc
```

Install dependencies

```bash
  cargo build --release
```

Encode any JSON file

```bash
  ./target/release/jsonc encode my-file.json
```

And decode any JSONB file

```bash
  ./target/release/jsonc decode my-file.jsonb
```

## Authors

- [@thomasvergne](https://www.github.com/thomasvergne)
