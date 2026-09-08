use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::ast::{Program, StmtKind};
use crate::error::MuninnError;
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::span::Span;
use crate::token::Token;
use crate::typecheck::{Reference, SemanticModel, Symbol, analyze_program, check_program};

#[derive(Debug, Clone, Default)]
pub struct FrontendAnalysis {
    pub parsed: Option<Program>,
    pub semantics: Option<SemanticModel>,
    pub diagnostics: Vec<MuninnError>,
}

impl FrontendAnalysis {
    pub fn is_ok(&self) -> bool {
        self.diagnostics.is_empty()
    }

    pub fn has_errors(&self) -> bool {
        !self.is_ok()
    }

    pub fn definition_at_offset(&self, offset: usize) -> Option<&Symbol> {
        self.semantics.as_ref()?.definition_at_offset(offset)
    }

    pub fn symbol_at_offset(&self, offset: usize) -> Option<&Symbol> {
        self.semantics.as_ref()?.symbol_at_offset(offset)
    }

    pub fn reference_at_offset(&self, offset: usize) -> Option<&Reference> {
        self.semantics.as_ref()?.reference_at_offset(offset)
    }
}

/// All spans naming symbol `target`: the definition span first (when it is a
/// real source span; native builtins carry a synthetic line-0 span), then
/// every reference span. Powers LSP references, rename, and document
/// highlight without text matching, so shadowed locals and same-named
/// symbols in other functions stay distinct.
pub fn references_to_target(analysis: &FrontendAnalysis, target: usize) -> Vec<Span> {
    let mut spans = Vec::new();
    let Some(semantics) = analysis.semantics.as_ref() else {
        return spans;
    };
    if let Some(symbol) = semantics.symbol_by_id(target)
        && symbol.span.line != 0
    {
        spans.push(symbol.span);
    }
    spans.extend(
        semantics
            .references
            .iter()
            .filter(|reference| reference.target == target)
            .map(|reference| reference.span),
    );
    spans
}

/// Returns true when `name` is a plain Muninn identifier and therefore a
/// valid rename target.
pub fn is_rename_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => (),
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

pub fn lex_document(source: &str) -> Result<Vec<Token>, Vec<MuninnError>> {
    Lexer::new(source).lex()
}

pub fn parse_document(source: &str) -> Result<Program, Vec<MuninnError>> {
    let tokens = lex_document(source)?;
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

pub fn check_document(program: &Program) -> Result<SemanticModel, Vec<MuninnError>> {
    check_program(program)
}

pub fn analyze_document(source: &str) -> FrontendAnalysis {
    let parsed = match parse_document(source) {
        Ok(program) => program,
        Err(diagnostics) => {
            return FrontendAnalysis {
                parsed: None,
                semantics: None,
                diagnostics,
            };
        }
    };

    let semantics = analyze_program(&parsed);
    let diagnostics = semantics.diagnostics.clone();
    FrontendAnalysis {
        parsed: Some(parsed),
        semantics: Some(semantics),
        diagnostics,
    }
}

/// One file participating in a loaded project.
#[derive(Debug, Clone)]
pub struct ProjectFile {
    pub path: PathBuf,
    pub source: String,
}

/// A diagnostic tied to the file that owns its span. Multi-file spans
/// render against the owning file: line and column are exact, offsets
/// stay in combined-source space and only feed multi-line markers.
#[derive(Debug, Clone)]
pub struct FileDiagnostic {
    pub path: PathBuf,
    pub source: String,
    pub error: MuninnError,
}

impl FileDiagnostic {
    pub fn render(&self) -> String {
        format!(
            "{}: {}",
            self.path.display(),
            self.error.render_with_source(&self.source)
        )
    }
}

/// An import-expanded project: dependency bodies first, entry last.
/// Bodies keep their internal line structure (import statements are
/// blanked, not removed), so per-file line numbers survive; only
/// inter-file offsets shift, and [`LoadedProject::map_errors`] reverses
/// that shift for diagnostics.
#[derive(Debug, Clone)]
pub struct LoadedProject {
    pub entry: PathBuf,
    pub files: Vec<ProjectFile>,
    pub combined: String,
    line_starts: Vec<usize>,
}

impl LoadedProject {
    fn owner(&self, line: usize) -> usize {
        let mut owner = 0;
        for (index, start) in self.line_starts.iter().enumerate() {
            if *start <= line {
                owner = index;
            } else {
                break;
            }
        }
        owner.min(self.files.len().saturating_sub(1))
    }

    /// Rewrites combined-source diagnostics to per-file diagnostics.
    pub fn map_errors(&self, errors: Vec<MuninnError>) -> Vec<FileDiagnostic> {
        errors
            .into_iter()
            .map(|mut error| {
                if error.span.line == 0 || self.files.is_empty() {
                    let fallback = self.files.first();
                    return FileDiagnostic {
                        path: fallback
                            .map(|file| file.path.clone())
                            .unwrap_or_else(|| self.entry.clone()),
                        source: fallback.map(|file| file.source.clone()).unwrap_or_default(),
                        error,
                    };
                }
                let owner = self.owner(error.span.line);
                let start = self.line_starts.get(owner).copied().unwrap_or(1);
                error.span.line = error.span.line.saturating_sub(start - 1).max(1);
                error.span.end_line = error.span.end_line.saturating_sub(start - 1).max(1);
                FileDiagnostic {
                    path: self.files[owner].path.clone(),
                    source: self.files[owner].source.clone(),
                    error,
                }
            })
            .collect()
    }
}

fn file_diagnostic(path: &Path, source: &str, error: MuninnError) -> FileDiagnostic {
    FileDiagnostic {
        path: path.to_path_buf(),
        source: source.to_string(),
        error,
    }
}

/// Validates one import specifier. Only `./` and `../` file-relative
/// paths with a `.mun` extension are accepted: absolute paths are
/// unportable, bare names invite search-path ambiguity, and `@` is
/// reserved for future package aliases.
fn check_specifier(spec: &str, span: Span) -> Result<(), MuninnError> {
    if spec.starts_with('@') {
        return Err(MuninnError::new(
            "parser",
            "'@' prefixes are reserved for future package aliases; use a './' or '../' relative path",
            span,
        ));
    }
    if Path::new(spec).is_absolute() {
        return Err(MuninnError::new(
            "parser",
            format!(
                "absolute import paths are not allowed: \"{spec}\" (use a './' or '../' path relative to the importing file)"
            ),
            span,
        ));
    }
    if !spec.starts_with("./") && !spec.starts_with("../") {
        return Err(MuninnError::new(
            "parser",
            format!(
                "import paths must be file-relative (start with './' or '../'), got \"{spec}\""
            ),
            span,
        ));
    }
    if !spec.ends_with(".mun") {
        return Err(MuninnError::new(
            "parser",
            format!("import path must end with '.mun', got \"{spec}\""),
            span,
        ));
    }
    Ok(())
}

struct Loader {
    files: Vec<ProjectFile>,
    bodies: Vec<String>,
    completed: HashSet<PathBuf>,
    stack: Vec<PathBuf>,
}

impl Loader {
    fn load(&mut self, path: &Path) -> Result<(), Vec<FileDiagnostic>> {
        let canonical = path.canonicalize().map_err(|error| {
            vec![FileDiagnostic {
                path: path.to_path_buf(),
                source: String::new(),
                error: MuninnError::new(
                    "parser",
                    format!("cannot load \"{}\": {error}", path.display()),
                    Span::default(),
                ),
            }]
        })?;
        if self.completed.contains(&canonical) {
            return Ok(());
        }
        if let Some(cycle_at) = self.stack.iter().position(|p| p == &canonical) {
            let mut chain: Vec<String> = self.stack[cycle_at..]
                .iter()
                .map(|p| file_label(p))
                .collect();
            chain.push(file_label(&canonical));
            let importer = self
                .stack
                .last()
                .cloned()
                .unwrap_or_else(|| canonical.clone());
            let (importer_source, import_span) = import_site_span(&importer, &canonical);
            return Err(vec![file_diagnostic(
                &importer,
                &importer_source,
                MuninnError::new(
                    "parser",
                    format!("circular import: {}", chain.join(" -> ")),
                    import_span,
                ),
            )]);
        }
        let source = std::fs::read_to_string(&canonical).map_err(|error| {
            vec![FileDiagnostic {
                path: canonical.clone(),
                source: String::new(),
                error: MuninnError::new(
                    "parser",
                    format!("cannot read \"{}\": {error}", canonical.display()),
                    Span::default(),
                ),
            }]
        })?;
        let program = parse_document(&source).map_err(|errors| {
            errors
                .into_iter()
                .map(|error| file_diagnostic(&canonical, &source, error))
                .collect::<Vec<_>>()
        })?;
        self.stack.push(canonical.clone());
        let mut imports = Vec::new();
        for statement in &program.statements {
            if let StmtKind::Import { path, span } = &statement.kind {
                imports.push((path.clone(), *span));
            }
        }
        for (spec, span) in &imports {
            if let Err(error) = check_specifier(spec, *span) {
                let failed = self.stack.pop();
                debug_assert!(failed.is_some());
                return Err(vec![file_diagnostic(&canonical, &source, error)]);
            }
            let parent = canonical.parent().unwrap_or_else(|| Path::new("."));
            let target = parent.join(spec);
            if let Err(errors) = self.load(&target) {
                // Nested failures already name their own files; only the
                // specifier error belongs to this import site.
                let failed = self.stack.pop();
                debug_assert!(failed.is_some());
                return Err(errors);
            }
        }
        self.stack.pop();
        // Blank import statements (newlines preserved) so the body keeps
        // this file's line structure in the combined source. No `unsafe`:
        // each non-newline char becomes spaces matching its byte length.
        let mut body = source.clone();
        for (_, span) in &imports {
            let start = span.offset.min(body.len());
            let end = span.end_offset.min(body.len()).max(start);
            if !body.is_char_boundary(start) || !body.is_char_boundary(end) {
                continue;
            }
            let blanked: String = body[start..end]
                .chars()
                .map(|ch| {
                    if ch == '\n' {
                        "\n".to_string()
                    } else {
                        " ".repeat(ch.len_utf8())
                    }
                })
                .collect();
            body.replace_range(start..end, &blanked);
        }
        self.completed.insert(canonical.clone());
        self.files.push(ProjectFile {
            path: canonical,
            source,
        });
        self.bodies.push(body);
        Ok(())
    }
}

/// Finds the import statement in `importer` that resolved to
/// `target`, for cycle diagnostics. Error paths only.
fn import_site_span(importer: &Path, target: &Path) -> (String, Span) {
    let source = std::fs::read_to_string(importer).unwrap_or_default();
    if let Ok(program) = parse_document(&source) {
        let parent = importer.parent().unwrap_or_else(|| Path::new("."));
        for statement in &program.statements {
            if let StmtKind::Import { path, span } = &statement.kind
                && check_specifier(path, *span).is_ok()
                && parent.join(path).canonicalize().ok().as_deref() == Some(target)
            {
                return (source, *span);
            }
        }
    }
    (source, Span::default())
}

fn file_label(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Loads an entry file plus its transitive `./`/`../` imports.
/// Dependencies come first, the entry last; cycles fail with the full
/// chain instead of recursing. Diamond imports load once.
pub fn load_project(entry: &Path) -> Result<LoadedProject, Vec<FileDiagnostic>> {
    let mut loader = Loader {
        files: Vec::new(),
        bodies: Vec::new(),
        completed: HashSet::new(),
        stack: Vec::new(),
    };
    // Pre-canonicalize the entry so the cycle chain shows real paths.
    let canonical_entry = entry.canonicalize().unwrap_or_else(|_| entry.to_path_buf());
    loader.load(entry)?;
    let mut combined = String::new();
    let mut line_starts = Vec::with_capacity(loader.files.len());
    for body in &loader.bodies {
        line_starts.push(combined.lines().count() + 1);
        combined.push_str(body);
        if !combined.ends_with('\n') {
            combined.push('\n');
        }
    }
    Ok(LoadedProject {
        entry: canonical_entry,
        files: loader.files,
        combined,
        line_starts,
    })
}
