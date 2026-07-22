use std::collections::{HashMap, HashSet};

use aho_corasick::{AhoCorasick, MatchKind};
use jsonc_mlir::MLIR;

/// String optimization structure definition
/// This stores the collected strings during the first phase of analysis and also
/// stores the strings occurrences during the second phase of analysis.
pub struct StringOptimizer<'a> {
    pub strings: Vec<&'a str>,
    string_counts: HashMap<&'a str, usize>,
}

impl<'a> StringOptimizer<'a> {
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            string_counts: HashMap::new(),
        }
    }

    /// Build a string expression from the given strings, formally:
    ///
    /// ```
    /// StringExpr = String | StringExpr + String
    /// ```
    ///
    /// Returns a [`MLIR::String`] if the vector is empty, a single string if the
    /// vector has a single element, or a [`MLIR::Add`] expression otherwise.
    pub fn build_string_expr(&self, strings: Vec<&'a str>) -> MLIR<'a> {
        let filtered: Vec<&'a str> = strings.into_iter().filter(|s| !s.is_empty()).collect();

        if filtered.is_empty() {
            return MLIR::String("");
        }
        if filtered.len() == 1 {
            return MLIR::String(filtered[0]);
        }

        filtered[1..]
            .iter()
            .fold(MLIR::String(filtered[0]), |acc, &s| MLIR::Add {
                left: Box::new(acc),
                right: Box::new(MLIR::String(s)),
            })
    }

    /// Adds a string to the string pool and increments its count in the string counts map.
    fn add_string(&mut self, s: &'a str) {
        self.strings.push(s);
        *self.string_counts.entry(s).or_insert(0) += 1;
    }

    /// Traverses the MLIR tree and collects all strings into the string pool.
    pub fn traverse_and_collect_strings(&mut self, mlir: &MLIR<'a>) {
        match mlir {
            MLIR::String(s) => self.add_string(s),
            MLIR::Array(arr) => {
                for e in arr {
                    self.traverse_and_collect_strings(e);
                }
            }
            MLIR::Object(obj) => {
                for (_, v) in obj {
                    self.traverse_and_collect_strings(v);
                }
            }
            _ => {}
        }
    }

    /// Removes all occurrences of `s` from the string pool and returns the remaining strings.
    pub fn remove_string(&self, s: &str) -> Vec<&'a str> {
        self.strings.iter().copied().filter(|&x| x != s).collect()
    }

    /// Decompose `s` into substrings from candidates using Aho-Corasick.
    /// Used to decompose a string into a sequence of substrings from the string pool,
    /// and to be later optimized into a [`MLIR::String`] or [`MLIR::Add`] expression
    /// using [`StringOptimizer::optimize_with_ac`].
    fn substrings_ac<'b>(
        s: &'a str,
        ac: &AhoCorasick,
        candidates: &'b [&'a str],
    ) -> Option<Vec<&'a str>> {
        if s.is_empty() {
            return None;
        }

        let mut result: Vec<&'a str> = Vec::new();
        let mut last_end: usize = 0;
        let mut has_match = false;

        for mat in ac.find_iter(s) {
            has_match = true;
            let gap = &s[last_end..mat.start()];
            if !gap.is_empty() {
                result.push(gap);
            }
            result.push(candidates[mat.pattern().as_usize()]);
            last_end = mat.end();
        }

        let tail = &s[last_end..];
        if !tail.is_empty() {
            result.push(tail);
        }

        if !has_match || (result.len() == 1 && result[0] == s) {
            None
        } else {
            Some(result)
        }
    }

    /// Main optimization function that uses Aho-Corasick to decompose a string
    /// into a sequence of substrings from the string pool, and then builds
    /// a [`MLIR::String`] or [`MLIR::Add`] expression from the parts.
    fn optimize_with_ac(
        &self,
        mlir: &MLIR<'a>,
        ac: &AhoCorasick,
        candidates: &[&'a str],
        candidates_set: &HashSet<&'a str>,
        min_candidate_len: usize,
    ) -> MLIR<'a> {
        match mlir {
            MLIR::String(s) if !s.is_empty() => {
                if s.len() < min_candidate_len {
                    MLIR::String(s)
                } else if candidates_set.contains(s) {
                    MLIR::String(s)
                } else if let Some(parts) = Self::substrings_ac(s, ac, candidates) {
                    self.build_string_expr(parts)
                } else {
                    MLIR::String(s)
                }
            }
            MLIR::Array(arr) => MLIR::Array(
                arr.iter()
                    .map(|e| {
                        self.optimize_with_ac(e, ac, candidates, candidates_set, min_candidate_len)
                    })
                    .collect(),
            ),
            MLIR::Object(obj) => MLIR::Object(
                obj.iter()
                    .map(|(k, v)| {
                        (
                            *k,
                            self.optimize_with_ac(
                                v,
                                ac,
                                candidates,
                                candidates_set,
                                min_candidate_len,
                            ),
                        )
                    })
                    .collect(),
            ),
            _ => mlir.clone(),
        }
    }

    /// Main optimization function that uses Aho-Corasick to decompose a string
    /// into a sequence of substrings from the string pool, and then builds
    /// a [`MLIR::String`] or [`MLIR::Add`] expression from the parts.
    ///
    /// This function defines the prerequisites for string optimization:
    /// - The string must be non-empty
    /// - The string must occur at least `MIN_CANDIDATE_OCCURRENCES` times
    /// - The string must be at least `MIN_CANDIDATE_LEN` characters long
    pub fn optimize(&mut self, mlir: &MLIR<'a>) -> MLIR<'a> {
        const MIN_CANDIDATE_OCCURRENCES: usize = 2;
        const MIN_CANDIDATE_LEN: usize = 128;

        let mut candidates: Vec<&'a str> = self
            .string_counts
            .iter()
            .filter_map(|(s, count)| {
                if !s.is_empty()
                    && *count >= MIN_CANDIDATE_OCCURRENCES
                    && s.len() >= MIN_CANDIDATE_LEN
                {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        candidates.sort_unstable_by(|a, b| b.len().cmp(&a.len()));

        if candidates.is_empty() {
            return mlir.clone();
        }
        let candidates_set: HashSet<&'a str> = candidates.iter().copied().collect();

        let ac = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&candidates)
            .expect("Aho-Corasick: construction failed");

        self.optimize_with_ac(mlir, &ac, &candidates, &candidates_set, MIN_CANDIDATE_LEN)
    }
}
