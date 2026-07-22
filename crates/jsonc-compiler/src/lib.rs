use std::collections::HashMap;

use jsonc_bytecode::{Address, Literal, OpCode};
use jsonc_mlir::{MLIR, debug};

/// The compiler struct that compiles JSONC source code into MLIR and bytecode.
///
/// It maintains a value pool, literal indices, and symbol tables for locals, globals, and functions.
/// Also maintains literal weights for string literals to prioritize common strings.
pub struct Compiler<'a> {
    pub value_pool: Vec<Literal>,
    literal_indices: HashMap<Literal, Address>,
    locals: Vec<&'a str>,
    local_indices: HashMap<&'a str, Address>,
    globals: Vec<&'a str>,
    global_indices: HashMap<&'a str, Address>,
    pub functions: HashMap<&'a str, Address>,
    pub literal_weights: HashMap<String, i64>,
}

impl<'a> Compiler<'a> {
    pub fn new() -> Self {
        Self {
            value_pool: vec![Literal::Null, Literal::Bool(false), Literal::Bool(true)],
            literal_indices: HashMap::from([
                (Literal::Null, 0),
                (Literal::Bool(false), 1),
                (Literal::Bool(true), 2),
            ]),
            locals: Vec::new(),
            local_indices: HashMap::new(),
            globals: Vec::new(),
            global_indices: HashMap::new(),
            functions: HashMap::new(),
            literal_weights: HashMap::new(),
        }
    }

    /// Interns a literal into the value pool and returns its address.
    fn intern_literal(&mut self, literal: Literal) -> Address {
        match self.literal_indices.entry(literal) {
            std::collections::hash_map::Entry::Occupied(e) => *e.get(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let index = self.value_pool.len() as Address;
                self.value_pool.push(e.key().clone());
                *e.insert(index)
            }
        }
    }

    /// Pushes a local variable name onto the local symbol table.
    fn push_local(&mut self, name: &'a str) {
        if self.local_indices.contains_key(name) {
            return;
        }
        let index = self.locals.len() as Address;
        self.locals.push(name);
        self.local_indices.insert(name, index);
    }

    /// Pushes a global variable name onto the global symbol table.
    fn push_global(&mut self, name: &'a str) {
        if self.global_indices.contains_key(name) {
            return;
        }
        let index = self.globals.len() as Address;
        self.globals.push(name);
        self.global_indices.insert(name, index);
    }

    /// Compiles all MLIR nodes into bytecode.
    ///
    /// This method pre-registers all top-level functions and globals, then compiles each node into bytecode.
    pub fn compile_all(&mut self, mlir: Vec<MLIR<'a>>) -> Vec<OpCode> {
        // Pre-register all top-level functions and globals
        for node in &mlir {
            match node {
                MLIR::MakeFunction { name, .. } => {
                    self.push_global(*name);
                }
                MLIR::Let { name, .. } => {
                    self.push_global(*name);
                }
                _ => {}
            }
        }

        let mut instructions = Vec::with_capacity(mlir.len() * 4);

        for node in &mlir {
            self.compile_into(node, false, &mut instructions);
        }

        self.remap_globals_by_frequency(&mut instructions);
        self.remap_literals_by_frequency(&mut instructions);

        #[cfg(debug_assertions)]
        let mut num_numeric = 0;
        #[cfg(debug_assertions)]
        let mut num_timestamp = 0;
        #[cfg(debug_assertions)]
        let mut num_other_str = 0;
        #[cfg(debug_assertions)]
        let mut bytes_other_str = 0;
        #[cfg(debug_assertions)]
        let mut num_other = 0;

        #[cfg(debug_assertions)]
        let is_numeric_string = |s: &str| -> bool {
            if s.is_empty() || s.len() > 20 {
                return false;
            }
            s.chars().all(|c| c.is_ascii_digit()) && s.parse::<u64>().is_ok()
        };

        #[cfg(debug_assertions)]
        let is_timestamp_string = |s: &str| -> bool {
            if s.len() != 32 {
                return false;
            }
            let bytes = s.as_bytes();
            bytes[4] == b'-'
                && bytes[7] == b'-'
                && bytes[10] == b'T'
                && bytes[13] == b':'
                && bytes[16] == b':'
                && bytes[19] == b'.'
                && bytes[26] == b'+'
                && bytes[29] == b':'
                && &s[27..32] == "00:00"
        };

        #[cfg(debug_assertions)]
        for lit in &self.value_pool {
            match lit {
                Literal::String(s) => {
                    if is_numeric_string(s) {
                        num_numeric += 1;
                    } else if is_timestamp_string(s) {
                        num_timestamp += 1;
                    } else {
                        num_other_str += 1;
                        bytes_other_str += s.len();
                    }
                }
                _ => num_other += 1,
            }
        }

        debug!("[DEBUG_LITS] Numeric strings: {}", num_numeric);
        debug!("[DEBUG_LITS] Timestamp strings: {}", num_timestamp);
        debug!(
            "[DEBUG_LITS] Other strings count: {}, bytes: {}",
            num_other_str, bytes_other_str
        );
        debug!("[DEBUG_LITS] Non-string literals: {}", num_other);

        instructions
    }

    /// Remaps global variable indices by frequency to reduce instruction size.
    ///
    /// This method replaces global variable indices with their frequency-based index to reduce instruction size.
    fn remap_globals_by_frequency(&mut self, instructions: &mut [OpCode]) {
        if self.globals.is_empty() {
            return;
        }

        let mut frequencies = vec![0usize; self.globals.len()];
        for instr in instructions.iter() {
            match instr {
                OpCode::LoadGlobal { var_index } | OpCode::StoreGlobal { var_index } => {
                    if let Some(freq) = frequencies.get_mut(*var_index as usize) {
                        *freq += 1;
                    }
                }
                OpCode::MakeFieldGlobalBlock { fields } => {
                    for (_, addr) in fields {
                        if let Some(freq) = frequencies.get_mut(*addr as usize) {
                            *freq += 1;
                        }
                    }
                }
                OpCode::MakeFieldFromGlobal { addr, .. } => {
                    if let Some(freq) = frequencies.get_mut(*addr as usize) {
                        *freq += 1;
                    }
                }
                OpCode::AddGlobalLeft { addr } | OpCode::AddGlobalRight { addr } => {
                    if let Some(freq) = frequencies.get_mut(*addr as usize) {
                        *freq += 1;
                    }
                }
                OpCode::AddGlobal2 { addr1, addr2 } => {
                    if let Some(freq) = frequencies.get_mut(*addr1 as usize) {
                        *freq += 1;
                    }
                    if let Some(freq) = frequencies.get_mut(*addr2 as usize) {
                        *freq += 1;
                    }
                }
                _ => {}
            }
        }

        let mut old_indices = (0..self.globals.len()).collect::<Vec<usize>>();
        old_indices.sort_by(|a, b| frequencies[*b].cmp(&frequencies[*a]).then_with(|| a.cmp(b)));

        let mut remap = vec![0usize; self.globals.len()];
        for (new_index, old_index) in old_indices.iter().enumerate() {
            remap[*old_index] = new_index;
        }

        if remap.iter().enumerate().all(|(idx, mapped)| idx == *mapped) {
            return;
        }

        for instr in instructions.iter_mut() {
            match instr {
                OpCode::LoadGlobal { var_index } | OpCode::StoreGlobal { var_index } => {
                    *var_index = remap[*var_index as usize] as Address;
                }
                OpCode::MakeFieldGlobalBlock { fields } => {
                    for (_, addr) in fields {
                        *addr = remap[*addr as usize] as Address;
                    }
                }
                OpCode::MakeFieldFromGlobal { addr, .. } => {
                    *addr = remap[*addr as usize] as Address;
                }
                OpCode::AddGlobalLeft { addr } | OpCode::AddGlobalRight { addr } => {
                    *addr = remap[*addr as usize] as Address;
                }
                OpCode::AddGlobal2 { addr1, addr2 } => {
                    *addr1 = remap[*addr1 as usize] as Address;
                    *addr2 = remap[*addr2 as usize] as Address;
                }
                _ => {}
            }
        }

        let old_globals = self.globals.clone();
        self.globals = old_indices
            .iter()
            .map(|old_index| old_globals[*old_index])
            .collect();

        self.global_indices.clear();
        for (index, name) in self.globals.iter().enumerate() {
            self.global_indices.insert(*name, index as Address);
        }
    }

    /// Remaps literal indices by frequency to reduce instruction size.
    ///
    /// This method replaces literal indices with their frequency-based index to reduce instruction size.
    fn remap_literals_by_frequency(&mut self, instructions: &mut [OpCode]) {
        if self.value_pool.is_empty() {
            return;
        }

        let mut frequencies = vec![0usize; self.value_pool.len()];
        let mut can_inline = vec![false; self.value_pool.len()];
        for (i, lit) in self.value_pool.iter().enumerate() {
            if let Literal::String(_) = lit {
                can_inline[i] = true;
            }
        }

        for instr in instructions.iter() {
            match instr {
                OpCode::MakeString { value } => {
                    if let Some(freq) = frequencies.get_mut(*value as usize) {
                        *freq += 1;
                    }
                }
                OpCode::MakeInteger { value }
                | OpCode::MakeFloat { value }
                | OpCode::MakeField { field_name: value }
                | OpCode::AddAddr { addr: value }
                | OpCode::MakeFieldFromGlobal {
                    field_name: value, ..
                }
                | OpCode::MakeFieldFromLocal {
                    field_name: value, ..
                }
                | OpCode::MakeFieldFromIntegerImm {
                    field_name: value, ..
                }
                | OpCode::MakeFieldFromBoolean {
                    field_name: value, ..
                } => {
                    if let Some(freq) = frequencies.get_mut(*value as usize) {
                        *freq += 1;
                    }
                    if (*value as usize) < can_inline.len() {
                        can_inline[*value as usize] = false;
                    }
                }
                OpCode::AddAddr2 { addr1, addr2 } | OpCode::MakeArray2 { addr1, addr2 } => {
                    if let Some(freq) = frequencies.get_mut(*addr1 as usize) {
                        *freq += 1;
                    }
                    if let Some(freq) = frequencies.get_mut(*addr2 as usize) {
                        *freq += 1;
                    }
                    if (*addr1 as usize) < can_inline.len() {
                        can_inline[*addr1 as usize] = false;
                    }
                    if (*addr2 as usize) < can_inline.len() {
                        can_inline[*addr2 as usize] = false;
                    }
                }
                OpCode::MakeFieldFromAddr { field_name, addr } => {
                    if let Some(freq) = frequencies.get_mut(*field_name as usize) {
                        *freq += 1;
                    }
                    if let Some(freq) = frequencies.get_mut(*addr as usize) {
                        *freq += 1;
                    }
                    if (*field_name as usize) < can_inline.len() {
                        can_inline[*field_name as usize] = false;
                    }
                    if (*addr as usize) < can_inline.len() {
                        can_inline[*addr as usize] = false;
                    }
                }
                OpCode::MakePairArray { pairs } => {
                    for (addr1, addr2) in pairs {
                        if let Some(freq) = frequencies.get_mut(*addr1 as usize) {
                            *freq += 1;
                        }
                        if let Some(freq) = frequencies.get_mut(*addr2 as usize) {
                            *freq += 1;
                        }
                        if (*addr1 as usize) < can_inline.len() {
                            can_inline[*addr1 as usize] = false;
                        }
                        if (*addr2 as usize) < can_inline.len() {
                            can_inline[*addr2 as usize] = false;
                        }
                    }
                }
                OpCode::MakeFieldIntegerBlock { fields } => {
                    for (field_name, _) in fields {
                        if let Some(freq) = frequencies.get_mut(*field_name as usize) {
                            *freq += 1;
                        }
                        if (*field_name as usize) < can_inline.len() {
                            can_inline[*field_name as usize] = false;
                        }
                    }
                }
                OpCode::MakeFieldIntegerBlockSequential {
                    count,
                    start_field_address,
                    ..
                } => {
                    for addr in *start_field_address..(*start_field_address + *count) {
                        if let Some(freq) = frequencies.get_mut(addr as usize) {
                            *freq += 1;
                        }
                        if (addr as usize) < can_inline.len() {
                            can_inline[addr as usize] = false;
                        }
                    }
                }
                OpCode::MakeFieldGlobalBlock { fields } => {
                    for (field_name, _) in fields {
                        if let Some(freq) = frequencies.get_mut(*field_name as usize) {
                            *freq += 1;
                        }
                        if (*field_name as usize) < can_inline.len() {
                            can_inline[*field_name as usize] = false;
                        }
                    }
                }
                OpCode::MakeFieldAddrBlock { fields } => {
                    for (field_name, addr) in fields {
                        if let Some(freq) = frequencies.get_mut(*field_name as usize) {
                            *freq += 1;
                        }
                        if let Some(freq) = frequencies.get_mut(*addr as usize) {
                            *freq += 1;
                        }
                        if (*field_name as usize) < can_inline.len() {
                            can_inline[*field_name as usize] = false;
                        }
                        if (*addr as usize) < can_inline.len() {
                            can_inline[*addr as usize] = false;
                        }
                    }
                }
                OpCode::MakeFieldBooleanBlock { fields } => {
                    for (field_name, _) in fields {
                        if let Some(freq) = frequencies.get_mut(*field_name as usize) {
                            *freq += 1;
                        }
                        if (*field_name as usize) < can_inline.len() {
                            can_inline[*field_name as usize] = false;
                        }
                    }
                }
                OpCode::MakeFieldLocalBlock { fields } => {
                    for (field_name, _) in fields {
                        if let Some(freq) = frequencies.get_mut(*field_name as usize) {
                            *freq += 1;
                        }
                        if (*field_name as usize) < can_inline.len() {
                            can_inline[*field_name as usize] = false;
                        }
                    }
                }
                _ => {}
            }
        }

        let mut inlined_strings = std::collections::HashMap::new();
        for i in 0..self.value_pool.len() {
            if frequencies[i] == 1 && can_inline[i] {
                if let Literal::String(s) = &self.value_pool[i] {
                    inlined_strings.insert(i as Address, s.clone());
                    frequencies[i] = 0;
                }
            }
        }

        // Checks if a string is a numeric string.
        //
        // Returns the parsed numeric value as a `u64` if the string is a valid numeric string, otherwise `None`.
        let is_numeric_string = |s: &str| -> Option<u64> {
            if s.is_empty() || s.len() > 20 {
                return None;
            }
            if !s.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            if let Ok(val) = s.parse::<u64>() {
                if val.to_string() == s {
                    return Some(val);
                }
            }
            None
        };

        // Checks if a string is a timestamp string in the format `YYYY-MM-DDTHH:MM:SS.sssZ`.
        //
        // Returns the parsed timestamp as a `u64` if the string is a valid timestamp, otherwise `None`.
        let is_timestamp_string = |s: &str| -> Option<u64> {
            if s.len() != 32 {
                return None;
            }
            let bytes = s.as_bytes();
            if bytes[4] != b'-' || bytes[7] != b'-' || bytes[10] != b'T' {
                return None;
            }
            if bytes[13] != b':' || bytes[16] != b':' || bytes[19] != b'.' {
                return None;
            }
            if bytes[26] != b'+' || bytes[29] != b':' {
                return None;
            }
            if &s[27..32] != "00:00" {
                return None;
            }
            let parse_digits = |start: usize, end: usize| -> Option<u64> {
                let mut val = 0u64;
                for i in start..end {
                    let digit = bytes[i];
                    if !digit.is_ascii_digit() {
                        return None;
                    }
                    val = val * 10 + (digit - b'0') as u64;
                }
                Some(val)
            };

            let year = parse_digits(0, 4)?;
            let month = parse_digits(5, 7)?;
            let day = parse_digits(8, 10)?;
            let hour = parse_digits(11, 13)?;
            let minute = parse_digits(14, 16)?;
            let second = parse_digits(17, 19)?;
            let microseconds = parse_digits(20, 26)?;

            if year < 2000 || year > 2255 {
                return None;
            }
            if month < 1 || month > 12 {
                return None;
            }
            if day < 1 || day > 31 {
                return None;
            }
            if hour > 23 || minute > 59 || second > 59 || microseconds > 999999 {
                return None;
            }

            let packed: u64 = (year - 2000) << 46
                | month << 42
                | day << 37
                | hour << 32
                | minute << 26
                | second << 20
                | microseconds;

            Some(packed)
        };

        for instr in instructions.iter_mut() {
            if let OpCode::MakeString { value } = instr
                && let Some(s) = inlined_strings.get(value)
            {
                if let Some(val) = is_numeric_string(s) {
                    *instr = OpCode::MakeStringNumInline { value: val };
                } else if let Some(val) = is_timestamp_string(s) {
                    *instr = OpCode::MakeStringTsInline { value: val };
                } else {
                    *instr = OpCode::MakeStringInline { value: s.clone() };
                }
            }
        }

        let is_numeric_literal = |lit: &Literal| -> bool {
            match lit {
                Literal::String(s) => {
                    is_numeric_string(s).is_some() || is_timestamp_string(s).is_some()
                }
                _ => true,
            }
        };

        let mut old_indices = (0..self.value_pool.len())
            .filter(|&i| frequencies[i] > 0)
            .collect::<Vec<usize>>();

        old_indices.sort_by(|a, b| {
            let a_is_str = !is_numeric_literal(&self.value_pool[*a]);
            let b_is_str = !is_numeric_literal(&self.value_pool[*b]);
            match (a_is_str, b_is_str) {
                (false, true) => std::cmp::Ordering::Less,
                (true, false) => std::cmp::Ordering::Greater,
                _ => {
                    if a_is_str {
                        let a_str = match &self.value_pool[*a] {
                            Literal::String(s) => s,
                            _ => unreachable!(),
                        };
                        let b_str = match &self.value_pool[*b] {
                            Literal::String(s) => s,
                            _ => unreachable!(),
                        };
                        let a_weight = self
                            .literal_weights
                            .get(a_str)
                            .copied()
                            .unwrap_or(1000000 + *a as i64);
                        let b_weight = self
                            .literal_weights
                            .get(b_str)
                            .copied()
                            .unwrap_or(1000000 + *b as i64);
                        a_weight.cmp(&b_weight)
                    } else {
                        frequencies[*b].cmp(&frequencies[*a]).then_with(|| a.cmp(b))
                    }
                }
            }
        });

        let mut remap = vec![0usize; self.value_pool.len()];
        for (new_index, old_index) in old_indices.iter().enumerate() {
            remap[*old_index] = new_index;
        }

        if remap.iter().enumerate().all(|(idx, mapped)| idx == *mapped) {
            return;
        }

        for instr in instructions.iter_mut() {
            match instr {
                OpCode::MakeInteger { value }
                | OpCode::MakeString { value }
                | OpCode::MakeFloat { value }
                | OpCode::MakeField { field_name: value }
                | OpCode::AddAddr { addr: value }
                | OpCode::MakeFieldFromGlobal {
                    field_name: value, ..
                }
                | OpCode::MakeFieldFromLocal {
                    field_name: value, ..
                }
                | OpCode::MakeFieldFromIntegerImm {
                    field_name: value, ..
                }
                | OpCode::MakeFieldFromBoolean {
                    field_name: value, ..
                } => {
                    *value = remap[*value as usize] as Address;
                }
                OpCode::AddAddr2 { addr1, addr2 } | OpCode::MakeArray2 { addr1, addr2 } => {
                    *addr1 = remap[*addr1 as usize] as Address;
                    *addr2 = remap[*addr2 as usize] as Address;
                }
                OpCode::MakeFieldFromAddr { field_name, addr } => {
                    *field_name = remap[*field_name as usize] as Address;
                    *addr = remap[*addr as usize] as Address;
                }
                OpCode::MakePairArray { pairs } => {
                    for (addr1, addr2) in pairs {
                        *addr1 = remap[*addr1 as usize] as Address;
                        *addr2 = remap[*addr2 as usize] as Address;
                    }
                }
                OpCode::MakeFieldIntegerBlock { fields } => {
                    for (field_name, _) in fields {
                        *field_name = remap[*field_name as usize] as Address;
                    }
                }
                OpCode::MakeFieldIntegerBlockSequential {
                    start_field_address,
                    ..
                } => {
                    *start_field_address = remap[*start_field_address as usize] as Address;
                }
                OpCode::MakeFieldGlobalBlock { fields } => {
                    for (field_name, _) in fields {
                        *field_name = remap[*field_name as usize] as Address;
                    }
                }
                OpCode::MakeFieldAddrBlock { fields } => {
                    for (field_name, addr) in fields {
                        *field_name = remap[*field_name as usize] as Address;
                        *addr = remap[*addr as usize] as Address;
                    }
                }
                OpCode::MakeFieldBooleanBlock { fields } => {
                    for (field_name, _) in fields {
                        *field_name = remap[*field_name as usize] as Address;
                    }
                }
                OpCode::MakeFieldLocalBlock { fields } => {
                    for (field_name, _) in fields {
                        *field_name = remap[*field_name as usize] as Address;
                    }
                }
                _ => {}
            }
        }

        let old_pool = self.value_pool.clone();
        self.value_pool = old_indices
            .iter()
            .map(|old_index| old_pool[*old_index].clone())
            .collect();

        self.literal_indices.clear();
        for (index, literal) in self.value_pool.iter().enumerate() {
            self.literal_indices
                .insert(literal.clone(), index as Address);
        }
    }

    /// Interns a literal value using the MLIR representation.
    fn intern_expr_literal(&mut self, mlir: &MLIR<'a>) -> Address {
        match mlir {
            MLIR::String(s) => self.intern_literal(Literal::String(s.to_string())),
            MLIR::Number(num) => self.intern_literal(Literal::Float(num.to_bits())),
            MLIR::Bool(b) => self.intern_literal(Literal::Bool(*b)),
            MLIR::Null => self.intern_literal(Literal::Null),
            _ => {
                unimplemented!()
            }
        }
    }

    /// Compiles an MLIR expression into bytecode instructions.
    fn compile_into(
        &mut self,
        mlir: &MLIR<'a>,
        is_in_function: bool,
        instructions: &mut Vec<OpCode>,
    ) {
        match mlir {
            MLIR::String(s) => {
                let index = self.intern_literal(Literal::String(s.to_string()));
                instructions.push(OpCode::MakeString { value: index });
            }

            MLIR::Number(num) => {
                if num.fract() == 0.0 && *num >= (i64::MIN as f64) && *num <= (i64::MAX as f64) {
                    instructions.push(OpCode::MakeIntegerImm { value: *num as i64 });
                } else {
                    let index = self.intern_literal(Literal::Float(num.to_bits()));
                    instructions.push(OpCode::MakeFloat { value: index });
                }
            }

            MLIR::Null => {
                instructions.push(OpCode::MakeNull);
            }

            MLIR::Bool(b) => {
                instructions.push(OpCode::MakeBoolean { value: *b });
            }

            MLIR::Array(arr) => {
                let len = arr.len();

                fn get_literal_value<'a>(mlir: &MLIR<'a>) -> Option<Literal> {
                    match mlir {
                        MLIR::String(s) => Some(Literal::String(s.to_string())),
                        MLIR::Number(num) => Some(Literal::Float(num.to_bits())),
                        MLIR::Bool(b) => Some(Literal::Bool(*b)),
                        MLIR::Null => Some(Literal::Null),
                        MLIR::Add { left, right } => {
                            let l = get_literal_value(left)?;
                            let r = get_literal_value(right)?;
                            if let (Literal::String(s1), Literal::String(s2)) = (l, r) {
                                Some(Literal::String(format!("{}{}", s1, s2)))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                }

                if len > 0
                    && arr.iter().all(|item| {
                        if let MLIR::Array(inner) = item
                            && inner.len() == 2
                            && get_literal_value(&inner[0]).is_some()
                            && get_literal_value(&inner[1]).is_some()
                        {
                            true
                        } else {
                            false
                        }
                    })
                {
                    let mut pairs = Vec::with_capacity(len);
                    for item in arr {
                        if let MLIR::Array(inner) = item {
                            let lit1 = get_literal_value(&inner[0]).unwrap();
                            let lit2 = get_literal_value(&inner[1]).unwrap();
                            let addr1 = self.intern_literal(lit1);
                            let addr2 = self.intern_literal(lit2);
                            pairs.push((addr1, addr2));
                        }
                    }
                    instructions.push(OpCode::MakePairArray { pairs });
                } else if len == 2
                    && let Some(lit1) = get_literal_value(&arr[0])
                    && let Some(lit2) = get_literal_value(&arr[1])
                {
                    let addr1 = self.intern_literal(lit1);
                    let addr2 = self.intern_literal(lit2);
                    instructions.push(OpCode::MakeArray2 { addr1, addr2 });
                } else {
                    for item in arr {
                        self.compile_into(item, is_in_function, instructions);
                    }
                    instructions.push(OpCode::MakeArray {
                        num_elements: len as u32,
                    });
                }
            }

            MLIR::Object(obj) => {
                let mut int_fields = Vec::new();
                let mut bool_fields = Vec::new();
                let mut global_fields = Vec::new();
                let mut local_fields = Vec::new();
                let mut addr_fields = Vec::new();

                let emit_blocks =
                    |int_fields: &mut Vec<(Address, i64)>,
                     bool_fields: &mut Vec<(Address, bool)>,
                     global_fields: &mut Vec<(Address, Address)>,
                     local_fields: &mut Vec<(Address, Address)>,
                     addr_fields: &mut Vec<(Address, Address)>,
                     instructions: &mut Vec<OpCode>| {
                        if !int_fields.is_empty() {
                            let sorted = std::mem::take(int_fields);
                            if sorted.len() == 1 {
                                instructions.push(OpCode::MakeFieldFromIntegerImm {
                                    field_name: sorted[0].0,
                                    value: sorted[0].1,
                                });
                            } else {
                                instructions.push(OpCode::MakeFieldIntegerBlock { fields: sorted });
                            }
                        }

                        if !bool_fields.is_empty() {
                            let fields = std::mem::take(bool_fields);
                            if fields.len() == 1 {
                                instructions.push(OpCode::MakeFieldFromBoolean {
                                    field_name: fields[0].0,
                                    value: fields[0].1,
                                });
                            } else {
                                instructions.push(OpCode::MakeFieldBooleanBlock { fields });
                            }
                        }

                        if !global_fields.is_empty() {
                            let fields = std::mem::take(global_fields);
                            if fields.len() == 1 {
                                instructions.push(OpCode::MakeFieldFromGlobal {
                                    field_name: fields[0].0,
                                    addr: fields[0].1,
                                });
                            } else {
                                instructions.push(OpCode::MakeFieldGlobalBlock { fields });
                            }
                        }

                        if !local_fields.is_empty() {
                            let fields = std::mem::take(local_fields);
                            if fields.len() == 1 {
                                instructions.push(OpCode::MakeFieldFromLocal {
                                    field_name: fields[0].0,
                                    addr: fields[0].1,
                                });
                            } else {
                                instructions.push(OpCode::MakeFieldLocalBlock { fields });
                            }
                        }

                        if !addr_fields.is_empty() {
                            let fields = std::mem::take(addr_fields);
                            if fields.len() == 1 {
                                instructions.push(OpCode::MakeFieldFromAddr {
                                    field_name: fields[0].0,
                                    addr: fields[0].1,
                                });
                            } else {
                                instructions.push(OpCode::MakeFieldAddrBlock { fields });
                            }
                        }
                    };

                for (key, value) in obj {
                    let index = self.intern_literal(Literal::String(key.to_string()));

                    if let MLIR::Number(num) = value
                        && num.fract() == 0.0
                        && *num >= (i64::MIN as f64)
                        && *num <= (i64::MAX as f64)
                    {
                        self.literal_weights.insert(key.to_string(), *num as i64);
                        int_fields.push((index, *num as i64));
                        continue;
                    }

                    if let MLIR::Bool(b) = value {
                        bool_fields.push((index, *b));
                        continue;
                    }

                    if let MLIR::Variable(n) = value
                        && let Some(value_addr) = self.global_indices.get(*n)
                    {
                        global_fields.push((index, *value_addr));
                        continue;
                    }

                    if let MLIR::Variable(n) = value
                        && let Some(value_addr) = self.local_indices.get(*n)
                    {
                        local_fields.push((index, *value_addr));
                        continue;
                    }

                    if matches!(
                        value,
                        MLIR::Number(_) | MLIR::String(_) | MLIR::Bool(_) | MLIR::Null
                    ) {
                        let value_addr = self.intern_expr_literal(value);
                        addr_fields.push((index, value_addr));
                        continue;
                    }

                    // A non-groupable element (nested object or array etc.)
                    // First emit accumulated blocks
                    emit_blocks(
                        &mut int_fields,
                        &mut bool_fields,
                        &mut global_fields,
                        &mut local_fields,
                        &mut addr_fields,
                        instructions,
                    );

                    self.compile_into(value, is_in_function, instructions);
                    instructions.push(OpCode::MakeField { field_name: index });
                }

                emit_blocks(
                    &mut int_fields,
                    &mut bool_fields,
                    &mut global_fields,
                    &mut local_fields,
                    &mut addr_fields,
                    instructions,
                );

                instructions.push(OpCode::MakeObject {
                    num_fields: obj.len() as u32,
                });
            }

            MLIR::Variable(name) => {
                if let Some(index) = self.local_indices.get(*name) {
                    instructions.push(OpCode::LoadLocal { var_index: *index });
                } else if let Some(index) = self.global_indices.get(*name) {
                    instructions.push(OpCode::LoadGlobal { var_index: *index });
                } else {
                    panic!("Variable '{}' not found", name);
                }
            }

            MLIR::MakeFunction { name, params, body } => {
                let prev_len = self.locals.len();

                for param in params {
                    self.push_local(*param);
                }

                let header_index = instructions.len();
                instructions.push(OpCode::MakeFunction {
                    num_params: params.len() as u8,
                    body_len: 0,
                });

                self.functions.insert(*name, instructions.len() as Address);
                let body_start = instructions.len();
                self.compile_into(body, true, instructions);
                let body_len = (instructions.len() - body_start) as u16;

                if let OpCode::MakeFunction {
                    body_len: header_body_len,
                    ..
                } = &mut instructions[header_index]
                {
                    *header_body_len = body_len;
                }

                for local_name in self.locals.drain(prev_len..) {
                    self.local_indices.remove(local_name);
                }
                self.push_global(*name);
            }

            MLIR::FunctionCall { name, args } => {
                let func_index = *self.functions.get(*name).expect(&format!(
                    "FunctionCall references an unknown function: {}",
                    name
                ));
                for arg in args {
                    self.compile_into(arg, is_in_function, instructions);
                }
                instructions.push(OpCode::CallFunction {
                    num_args: args.len() as u8,
                    func_index,
                });
            }

            MLIR::Add { left, right } => {
                if matches!(
                    **left,
                    MLIR::Number(_) | MLIR::String(_) | MLIR::Bool(_) | MLIR::Null
                ) && matches!(
                    **right,
                    MLIR::Number(_) | MLIR::String(_) | MLIR::Bool(_) | MLIR::Null
                ) {
                    let left_index = self.intern_expr_literal(left);
                    let right_index = self.intern_expr_literal(right);

                    instructions.push(OpCode::AddAddr2 {
                        addr1: left_index,
                        addr2: right_index,
                    });
                } else if matches!(
                    **left,
                    MLIR::Number(_) | MLIR::String(_) | MLIR::Bool(_) | MLIR::Null
                ) {
                    let left_index = self.intern_expr_literal(left);
                    self.compile_into(right, is_in_function, instructions);
                    instructions.push(OpCode::AddAddr { addr: left_index });
                } else if matches!(
                    **right,
                    MLIR::Number(_) | MLIR::String(_) | MLIR::Bool(_) | MLIR::Null
                ) {
                    let right_index = self.intern_expr_literal(right);
                    self.compile_into(left, is_in_function, instructions);
                    instructions.push(OpCode::AddAddr { addr: right_index });
                } else if let MLIR::Variable(n1) = &**left
                    && let MLIR::Variable(n2) = &**right
                    && !self.local_indices.contains_key(*n1)
                    && !self.local_indices.contains_key(*n2)
                {
                    let left_index = *self.global_indices.get(*n1).unwrap();
                    let right_index = *self.global_indices.get(*n2).unwrap();
                    instructions.push(OpCode::AddGlobal2 {
                        addr1: left_index,
                        addr2: right_index,
                    });
                } else if let MLIR::Variable(n1) = &**left
                    && !self.local_indices.contains_key(*n1)
                {
                    let left_index = *self.global_indices.get(*n1).unwrap();
                    self.compile_into(right, is_in_function, instructions);
                    instructions.push(OpCode::AddGlobalLeft { addr: left_index });
                } else if let MLIR::Variable(n2) = &**right
                    && !self.local_indices.contains_key(*n2)
                {
                    let right_index = *self.global_indices.get(*n2).unwrap();
                    self.compile_into(left, is_in_function, instructions);
                    instructions.push(OpCode::AddGlobalRight { addr: right_index });
                } else {
                    self.compile_into(left, is_in_function, instructions);
                    self.compile_into(right, is_in_function, instructions);
                    instructions.push(OpCode::Add);
                }
            }

            MLIR::Let { name, value } => {
                self.compile_into(value, is_in_function, instructions);

                let index = if is_in_function {
                    self.locals.len() as Address
                } else {
                    *self
                        .global_indices
                        .get(name)
                        .unwrap_or(&(self.globals.len() as Address))
                };

                instructions.push(if is_in_function {
                    OpCode::StoreLocal { var_index: index }
                } else {
                    OpCode::StoreGlobal { var_index: index }
                });

                if is_in_function {
                    self.push_local(*name);
                } else {
                    self.push_global(*name);
                }
            }
        }
    }

    /// Compiles a single MLIR node into bytecode instructions.
    pub fn compile(&mut self, mlir: &MLIR<'a>, is_in_function: bool) -> Vec<OpCode> {
        let mut instructions = Vec::new();
        self.compile_into(mlir, is_in_function, &mut instructions);

        instructions
    }
}
