use aho_corasick::{AhoCorasick, MatchKind};
use jsonc_mlir::MLIR;

pub struct StringOptimizer<'a> {
    pub strings: Vec<&'a str>,
}

impl<'a> StringOptimizer<'a> {
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
        }
    }

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

    fn add_string(&mut self, s: &'a str) {
        self.strings.push(s);
    }

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

    pub fn remove_string(&self, s: &str) -> Vec<&'a str> {
        self.strings.iter().copied().filter(|&x| x != s).collect()
    }

    /// Décompose `s` en sous-chaînes issues des candidats via Aho-Corasick.
    ///
    /// Complexité : O(N + M) par chaîne (N = len(s), M = total len des candidats)
    /// au lieu de O(C × N) avec la boucle précédente.
    /// Les espaces entre les correspondances sont retournés comme littéraux.
    fn substrings_ac<'b>(s: &'a str, ac: &AhoCorasick, candidates: &'b [&'a str]) -> Vec<&'a str> {
        if s.is_empty() {
            return Vec::new();
        }

        let mut result: Vec<&'a str> = Vec::new();
        let mut last_end: usize = 0;

        for mat in ac.find_iter(s) {
            // Littéral avant la correspondance (ne contient aucun candidat)
            let gap = &s[last_end..mat.start()];
            if !gap.is_empty() {
                result.push(gap);
            }
            // Candidat correspondant
            result.push(candidates[mat.pattern().as_usize()]);
            last_end = mat.end();
        }

        // Queue après la dernière correspondance
        let tail = &s[last_end..];
        if !tail.is_empty() {
            result.push(tail);
        }

        if result.is_empty() { vec![s] } else { result }
    }

    fn optimize_with_ac(
        &self,
        mlir: &MLIR<'a>,
        ac: &AhoCorasick,
        candidates: &[&'a str],
    ) -> MLIR<'a> {
        match mlir {
            MLIR::String(s) if !s.is_empty() => {
                let parts = Self::substrings_ac(s, ac, candidates);
                // Si la seule partie est la chaîne elle-même, rien à optimiser
                if parts.len() == 1 && parts[0] == *s {
                    MLIR::String(s)
                } else {
                    self.build_string_expr(parts)
                }
            }
            MLIR::Array(arr) => MLIR::Array(
                arr.iter()
                    .map(|e| self.optimize_with_ac(e, ac, candidates))
                    .collect(),
            ),
            MLIR::Object(obj) => MLIR::Object(
                obj.iter()
                    .map(|(k, v)| (*k, self.optimize_with_ac(v, ac, candidates)))
                    .collect(),
            ),
            _ => mlir.clone(),
        }
    }

    pub fn optimize(&mut self, mlir: &MLIR<'a>) -> MLIR<'a> {
        // Candidats triés du plus long au plus court, dédupliqués
        // (plus long en premier = correspondances plus greedy, meilleure compression)
        let mut candidates: Vec<&'a str> = self
            .strings
            .iter()
            .copied()
            .filter(|s| !s.is_empty())
            .collect();
        candidates.sort_unstable_by(|a, b| b.len().cmp(&a.len()));
        candidates.dedup();

        if candidates.is_empty() {
            return mlir.clone();
        }

        // Automate construit une seule fois pour tout l'arbre MLIR
        let ac = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&candidates)
            .expect("Aho-Corasick: construction echouee");

        self.optimize_with_ac(mlir, &ac, &candidates)
    }
}
