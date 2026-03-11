mod analysis;
mod convert;
mod state;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;

use analysis::{DefKind, SymbolDef, SymbolIndex, markdown_for_symbol};
use convert::{errors_to_diagnostics, span_to_range};
use muninn::analyze_document;
use muninn::span::Span;
use state::{DocumentState, ServerState, apply_content_changes};
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

static KEYWORD_COMPLETIONS: LazyLock<Vec<CompletionItem>> = LazyLock::new(|| {
    [
        "class", "fn", "let", "mut", "if", "else", "unless", "return", "while", "for", "in",
        "true", "false", "self",
    ]
    .into_iter()
    .map(|keyword| CompletionItem {
        label: keyword.to_string(),
        kind: Some(CompletionItemKind::KEYWORD),
        ..CompletionItem::default()
    })
    .collect()
});

static TYPE_COMPLETIONS: LazyLock<Vec<CompletionItem>> = LazyLock::new(|| {
    ["Int", "Float", "String", "Bool", "Void", "Option"]
        .into_iter()
        .map(|name| CompletionItem {
            label: name.to_string(),
            kind: Some(CompletionItemKind::CLASS),
            ..CompletionItem::default()
        })
        .collect()
});

static SEMANTIC_TOKEN_TYPES: &[SemanticTokenType] = &[
    SemanticTokenType::CLASS,
    SemanticTokenType::FUNCTION,
    SemanticTokenType::METHOD,
    SemanticTokenType::PROPERTY,
    SemanticTokenType::VARIABLE,
    SemanticTokenType::PARAMETER,
    SemanticTokenType::TYPE,
];

static SEMANTIC_TOKEN_MODIFIERS: &[SemanticTokenModifier] = &[SemanticTokenModifier::DECLARATION];

#[derive(Clone)]
struct Backend {
    client: Client,
    state: Arc<RwLock<ServerState>>,
}

impl Backend {
    async fn seed_document(&self, uri: Url, version: i32, source: String) {
        let mut state = self.state.write().await;
        state.docs.insert(
            uri,
            Arc::new(DocumentState::new(version, source, None, Vec::new())),
        );
    }

    async fn analyze_and_publish(&self, uri: Url, version: i32, source: String) {
        let frontend = analyze_document(&source);
        let symbols = frontend.parsed.as_ref().map(SymbolIndex::build);
        let document_symbols = symbols
            .as_ref()
            .map(build_document_symbols)
            .unwrap_or_default();
        let diagnostics = errors_to_diagnostics(&uri, &frontend.diagnostics);

        {
            let mut state = self.state.write().await;
            if let Some(existing) = state.docs.get(&uri)
                && existing.version > version
            {
                return;
            }

            state.docs.insert(
                uri.clone(),
                Arc::new(DocumentState::new(
                    version,
                    source,
                    symbols,
                    document_symbols,
                )),
            );
        }

        self.client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }

    fn schedule_analysis(&self, uri: Url, version: i32, source: String) {
        let backend = self.clone();
        tokio::spawn(async move {
            backend.analyze_and_publish(uri, version, source).await;
        });
    }

    async fn document(&self, uri: &Url) -> Option<Arc<DocumentState>> {
        self.state.read().await.docs.get(uri).cloned()
    }

    async fn apply_incremental_change(
        &self,
        uri: Url,
        version: i32,
        changes: &[TextDocumentContentChangeEvent],
    ) -> Option<String> {
        let current = self.document(&uri).await;
        let source = if let Some(current) = current {
            apply_content_changes(&current.source, changes)?
        } else {
            let first = changes.first()?;
            if first.range.is_some() {
                return None;
            }
            first.text.clone()
        };

        self.seed_document(uri, version, source.clone()).await;
        Some(source)
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        let capabilities = ServerCapabilities {
            text_document_sync: Some(TextDocumentSyncCapability::Kind(
                TextDocumentSyncKind::INCREMENTAL,
            )),
            hover_provider: Some(HoverProviderCapability::Simple(true)),
            definition_provider: Some(OneOf::Left(true)),
            document_symbol_provider: Some(OneOf::Left(true)),
            references_provider: Some(OneOf::Left(true)),
            rename_provider: Some(OneOf::Left(true)),
            workspace_symbol_provider: Some(OneOf::Left(true)),
            completion_provider: Some(CompletionOptions {
                resolve_provider: Some(false),
                trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
                all_commit_characters: None,
                work_done_progress_options: WorkDoneProgressOptions::default(),
                completion_item: None,
            }),
            signature_help_provider: Some(SignatureHelpOptions {
                trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
                retrigger_characters: Some(vec![",".to_string()]),
                work_done_progress_options: WorkDoneProgressOptions::default(),
            }),
            semantic_tokens_provider: Some(
                SemanticTokensServerCapabilities::SemanticTokensOptions(SemanticTokensOptions {
                    work_done_progress_options: WorkDoneProgressOptions::default(),
                    legend: SemanticTokensLegend {
                        token_types: SEMANTIC_TOKEN_TYPES.to_vec(),
                        token_modifiers: SEMANTIC_TOKEN_MODIFIERS.to_vec(),
                    },
                    range: Some(false),
                    full: Some(SemanticTokensFullOptions::Bool(true)),
                }),
            ),
            code_action_provider: Some(CodeActionProviderCapability::Simple(true)),
            ..ServerCapabilities::default()
        };

        Ok(InitializeResult {
            capabilities,
            server_info: Some(ServerInfo {
                name: "muninn-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Muninn LSP initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;
        let source = params.text_document.text;
        self.seed_document(uri.clone(), version, source.clone())
            .await;
        self.schedule_analysis(uri, version, source);
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;
        let Some(source) = self
            .apply_incremental_change(uri.clone(), version, &params.content_changes)
            .await
        else {
            self.client
                .log_message(
                    MessageType::WARNING,
                    "failed to apply incremental changes; waiting for full sync",
                )
                .await;
            return;
        };

        self.schedule_analysis(uri, version, source);
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        {
            let mut state = self.state.write().await;
            state.docs.remove(&params.text_document.uri);
        }
        self.client
            .publish_diagnostics(params.text_document.uri, Vec::new(), None)
            .await;
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let Some(doc) = self.document(&uri).await else {
            return Ok(None);
        };
        let Some(symbols) = &doc.symbols else {
            return Ok(None);
        };

        let offset = doc.offset_at(position.line, position.character);
        let reference = symbols.reference_at_offset(offset);
        let symbol = if let Some(reference) = reference {
            reference
                .target
                .and_then(|id| symbols.symbol_by_id(id))
                .or_else(|| symbols.symbol_at_offset(offset))
        } else {
            symbols.symbol_at_offset(offset)
        };

        let Some(symbol) = symbol else {
            return Ok(None);
        };

        let range = reference
            .map(|reference| span_to_range(reference.span))
            .unwrap_or_else(|| span_to_range(symbol.span));

        Ok(Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: markdown_for_symbol(symbol),
            }),
            range: Some(range),
        }))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let Some(doc) = self.document(&uri).await else {
            return Ok(None);
        };
        let Some(symbols) = &doc.symbols else {
            return Ok(None);
        };

        let offset = doc.offset_at(position.line, position.character);
        let Some(definition) = symbols.definition_at_offset(offset) else {
            return Ok(None);
        };

        if definition.span.line == 0 {
            return Ok(None);
        }

        let location = Location {
            uri,
            range: span_to_range(definition.span),
        };
        Ok(Some(GotoDefinitionResponse::Scalar(location)))
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let include_decl = params.context.include_declaration;

        let Some(doc) = self.document(&uri).await else {
            return Ok(None);
        };
        let Some(symbols) = &doc.symbols else {
            return Ok(None);
        };

        let offset = doc.offset_at(position.line, position.character);
        let Some(definition) = symbols.definition_at_offset(offset) else {
            return Ok(None);
        };

        let locations = symbols
            .references_for_target(definition.id, include_decl)
            .into_iter()
            .map(|span| Location {
                uri: uri.clone(),
                range: span_to_range(span),
            })
            .collect::<Vec<_>>();

        Ok(Some(locations))
    }

    async fn rename(&self, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
        let uri = params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let new_name = params.new_name;

        let Some(doc) = self.document(&uri).await else {
            return Ok(None);
        };
        let Some(symbols) = &doc.symbols else {
            return Ok(None);
        };

        let offset = doc.offset_at(position.line, position.character);
        let Some(definition) = symbols.definition_at_offset(offset) else {
            return Ok(None);
        };

        if matches!(definition.kind, DefKind::Builtin) {
            return Ok(None);
        }

        let edits = symbols
            .references_for_target(definition.id, true)
            .into_iter()
            .map(|span| TextEdit {
                range: span_to_range(span),
                new_text: new_name.clone(),
            })
            .collect::<Vec<_>>();

        let mut changes = HashMap::<Url, Vec<TextEdit>>::new();
        changes.insert(uri, edits);
        Ok(Some(WorkspaceEdit {
            changes: Some(changes),
            document_changes: None,
            change_annotations: None,
        }))
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let Some(doc) = self.document(&params.text_document.uri).await else {
            return Ok(None);
        };
        Ok(Some(DocumentSymbolResponse::Nested(
            doc.document_symbols.clone(),
        )))
    }

    #[allow(deprecated)]
    async fn symbol(
        &self,
        params: WorkspaceSymbolParams,
    ) -> Result<Option<Vec<SymbolInformation>>> {
        let query = params.query;
        let state = self.state.read().await;
        let mut items = Vec::<SymbolInformation>::new();
        for (uri, document) in &state.docs {
            let Some(symbols) = &document.symbols else {
                continue;
            };
            for symbol in symbols.search_symbols(&query) {
                items.push(SymbolInformation {
                    name: symbol.name.clone(),
                    kind: to_lsp_symbol_kind(symbol.kind),
                    tags: None,
                    deprecated: None,
                    location: Location {
                        uri: uri.clone(),
                        range: span_to_range(symbol.span),
                    },
                    container_name: None,
                });
            }
        }

        Ok(Some(items))
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let Some(doc) = self.document(&uri).await else {
            return Ok(None);
        };
        let Some(symbols) = &doc.symbols else {
            return Ok(Some(CompletionResponse::Array(KEYWORD_COMPLETIONS.clone())));
        };

        let offset = doc.offset_at(position.line, position.character);
        let (prefix_start, prefix_end) = doc.line_prefix_bounds(position.line, position.character);
        let prefix = &doc.source[prefix_start..prefix_end];

        if is_type_annotation_context(prefix) {
            let mut items = TYPE_COMPLETIONS.clone();
            for symbol in &symbols.defs {
                if symbol.kind == DefKind::Class {
                    items.push(symbol_to_completion(symbol));
                }
            }
            items.sort_by(|a, b| a.label.cmp(&b.label));
            return Ok(Some(CompletionResponse::Array(items)));
        }

        if let Some(chain) = receiver_chain_before_dot(prefix)
            && let Some(members) = symbols.resolve_member_chain(chain, offset)
        {
            let items = members
                .into_iter()
                .map(symbol_to_completion)
                .collect::<Vec<_>>();
            return Ok(Some(CompletionResponse::Array(items)));
        }

        let mut seen = std::collections::HashSet::<String>::new();
        let mut items = Vec::<CompletionItem>::new();

        for item in KEYWORD_COMPLETIONS.iter() {
            seen.insert(item.label.clone());
            items.push(item.clone());
        }

        for symbol in symbols.visible_symbols_before(offset) {
            if seen.insert(symbol.name.clone()) {
                items.push(symbol_to_completion(symbol));
            }
        }

        items.sort_by(|a, b| a.label.cmp(&b.label));
        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn signature_help(&self, params: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let Some(doc) = self.document(&uri).await else {
            return Ok(None);
        };
        let Some(symbols) = &doc.symbols else {
            return Ok(None);
        };

        let offset = doc.offset_at(position.line, position.character);
        let (start, end) = doc.line_prefix_bounds(position.line, position.character);
        let prefix = &doc.source[start..end];
        let Some((callee, active_param)) = call_context(prefix) else {
            return Ok(None);
        };

        let symbol = resolve_callee_symbol(symbols, callee, offset);
        let Some(symbol) = symbol else {
            return Ok(None);
        };

        let signature = SignatureInformation {
            label: if symbol.detail.is_empty() {
                symbol.name.clone()
            } else {
                symbol.detail.clone()
            },
            documentation: Some(Documentation::MarkupContent(MarkupContent {
                kind: MarkupKind::Markdown,
                value: markdown_for_symbol(symbol),
            })),
            parameters: parse_signature_parameters(&symbol.detail),
            active_parameter: None,
        };

        Ok(Some(SignatureHelp {
            signatures: vec![signature],
            active_signature: Some(0),
            active_parameter: Some(active_param as u32),
        }))
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = params.text_document.uri;
        let Some(doc) = self.document(&uri).await else {
            return Ok(None);
        };
        let Some(symbols) = &doc.symbols else {
            return Ok(None);
        };

        let mut entries = Vec::<(u32, u32, u32, u32, u32)>::new();

        for symbol in &symbols.defs {
            if symbol.span.line == 0 {
                continue;
            }
            if let Some((token_type, token_mod)) = semantic_for_def(symbol.kind) {
                entries.push(semantic_tuple(symbol.span, token_type, token_mod));
            }
        }

        for reference in &symbols.refs {
            let Some(target) = reference.target.and_then(|id| symbols.symbol_by_id(id)) else {
                continue;
            };
            if reference.span.line == 0 {
                continue;
            }
            if let Some((token_type, token_mod)) = semantic_for_ref(target.kind) {
                entries.push(semantic_tuple(reference.span, token_type, token_mod));
            }
        }

        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut data = Vec::<SemanticToken>::with_capacity(entries.len());
        let mut prev_line = 0u32;
        let mut prev_start = 0u32;

        for (line, start, len, token_type, token_modifiers_bitset) in entries {
            let delta_line = line.saturating_sub(prev_line);
            let delta_start = if delta_line == 0 {
                start.saturating_sub(prev_start)
            } else {
                start
            };
            data.push(SemanticToken {
                delta_line,
                delta_start,
                length: len.max(1),
                token_type,
                token_modifiers_bitset,
            });
            prev_line = line;
            prev_start = start;
        }

        Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data,
        })))
    }

    async fn code_action(&self, params: CodeActionParams) -> Result<Option<CodeActionResponse>> {
        let uri = params.text_document.uri;
        let mut actions = Vec::<CodeActionOrCommand>::new();

        for diagnostic in params.context.diagnostics {
            if diagnostic.message.contains("expected ';'") {
                let action = CodeAction {
                    title: "Insert missing ';'".to_string(),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![diagnostic.clone()]),
                    edit: Some(WorkspaceEdit {
                        changes: Some(HashMap::from([(
                            uri.clone(),
                            vec![TextEdit {
                                range: Range {
                                    start: diagnostic.range.end,
                                    end: diagnostic.range.end,
                                },
                                new_text: ";".to_string(),
                            }],
                        )])),
                        document_changes: None,
                        change_annotations: None,
                    }),
                    command: None,
                    is_preferred: Some(true),
                    disabled: None,
                    data: None,
                };
                actions.push(CodeActionOrCommand::CodeAction(action));
            }

            if let Some(name) = undefined_symbol_name(&diagnostic.message) {
                let action = CodeAction {
                    title: format!("Create 'let {} = 0;'", name),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![diagnostic.clone()]),
                    edit: Some(WorkspaceEdit {
                        changes: Some(HashMap::from([(
                            uri.clone(),
                            vec![TextEdit {
                                range: Range {
                                    start: Position {
                                        line: 0,
                                        character: 0,
                                    },
                                    end: Position {
                                        line: 0,
                                        character: 0,
                                    },
                                },
                                new_text: format!("let {} = 0;\n", name),
                            }],
                        )])),
                        document_changes: None,
                        change_annotations: None,
                    }),
                    command: None,
                    is_preferred: None,
                    disabled: None,
                    data: None,
                };
                actions.push(CodeActionOrCommand::CodeAction(action));
            }
        }

        Ok(Some(actions))
    }
}

fn build_document_symbols(symbols: &SymbolIndex) -> Vec<DocumentSymbol> {
    let mut by_parent = HashMap::<Option<usize>, Vec<&SymbolDef>>::new();
    for symbol in &symbols.defs {
        if symbol.span.line == 0 {
            continue;
        }
        by_parent.entry(symbol.container).or_default().push(symbol);
    }

    let top = by_parent.remove(&None).unwrap_or_default();
    let mut out = Vec::<DocumentSymbol>::new();
    for symbol in top {
        out.push(build_document_symbol(symbol, &by_parent));
    }
    out.sort_by_key(|symbol| (symbol.range.start.line, symbol.range.start.character));
    out
}

#[allow(deprecated)]
fn build_document_symbol(
    symbol: &SymbolDef,
    by_parent: &HashMap<Option<usize>, Vec<&SymbolDef>>,
) -> DocumentSymbol {
    let children = by_parent
        .get(&Some(symbol.id))
        .map(|items| {
            let mut nested = items
                .iter()
                .map(|child| build_document_symbol(child, by_parent))
                .collect::<Vec<_>>();
            nested.sort_by_key(|child| (child.range.start.line, child.range.start.character));
            nested
        })
        .unwrap_or_default();

    DocumentSymbol {
        name: symbol.name.clone(),
        detail: if symbol.detail.is_empty() {
            None
        } else {
            Some(symbol.detail.clone())
        },
        kind: to_lsp_symbol_kind(symbol.kind),
        tags: None,
        deprecated: None,
        range: span_to_range(symbol.span),
        selection_range: span_to_range(symbol.span),
        children: Some(children),
    }
}

fn to_lsp_symbol_kind(kind: DefKind) -> SymbolKind {
    match kind {
        DefKind::Variable => SymbolKind::VARIABLE,
        DefKind::Parameter => SymbolKind::VARIABLE,
        DefKind::Function => SymbolKind::FUNCTION,
        DefKind::Class => SymbolKind::CLASS,
        DefKind::Field => SymbolKind::FIELD,
        DefKind::Method => SymbolKind::METHOD,
        DefKind::Builtin => SymbolKind::FUNCTION,
    }
}

fn symbol_to_completion(symbol: &SymbolDef) -> CompletionItem {
    CompletionItem {
        label: symbol.name.clone(),
        kind: Some(match symbol.kind {
            DefKind::Variable | DefKind::Parameter => CompletionItemKind::VARIABLE,
            DefKind::Function | DefKind::Builtin => CompletionItemKind::FUNCTION,
            DefKind::Class => CompletionItemKind::CLASS,
            DefKind::Field => CompletionItemKind::FIELD,
            DefKind::Method => CompletionItemKind::METHOD,
        }),
        detail: if symbol.detail.is_empty() {
            None
        } else {
            Some(symbol.detail.clone())
        },
        ..CompletionItem::default()
    }
}

fn is_type_annotation_context(prefix: &str) -> bool {
    let trimmed = prefix.trim_end();
    trimmed.ends_with(':') || trimmed.ends_with("->") || trimmed.ends_with("Option[")
}

fn receiver_chain_before_dot(prefix: &str) -> Option<&str> {
    let trimmed = prefix.trim_end();
    if !trimmed.ends_with('.') {
        return None;
    }

    let end = trimmed.len().saturating_sub(1);
    let bytes = trimmed.as_bytes();
    let mut start = end;
    while start > 0 {
        let ch = bytes[start - 1] as char;
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' {
            start -= 1;
        } else {
            break;
        }
    }

    let chain = &trimmed[start..end];
    if chain.is_empty() || chain.starts_with('.') || chain.ends_with('.') {
        None
    } else {
        Some(chain)
    }
}

fn call_context(prefix: &str) -> Option<(&str, usize)> {
    let mut depth = 0usize;
    let mut call_start = None;
    for (index, ch) in prefix.char_indices().rev() {
        match ch {
            ')' => depth = depth.saturating_add(1),
            '(' => {
                if depth == 0 {
                    call_start = Some(index);
                    break;
                }
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
    }

    let open = call_start?;
    let before = prefix[..open].trim_end();
    let mut start = before.len();
    let bytes = before.as_bytes();
    while start > 0 {
        let ch = bytes[start - 1] as char;
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' {
            start -= 1;
        } else {
            break;
        }
    }
    let callee = &before[start..];
    if callee.is_empty() {
        return None;
    }

    let args = &prefix[open + 1..];
    let active_param = args.chars().filter(|ch| *ch == ',').count();
    Some((callee, active_param))
}

fn resolve_callee_symbol<'a>(
    symbols: &'a SymbolIndex,
    callee: &str,
    offset: usize,
) -> Option<&'a SymbolDef> {
    if let Some((receiver, member)) = callee.rsplit_once('.') {
        let class_name = symbols.class_for_name_before(receiver, offset)?;
        let members = symbols.class_members.get(&class_name)?;
        let id = *members.get(member)?;
        return symbols.symbol_by_id(id);
    }

    symbols
        .visible_symbols_before(offset)
        .into_iter()
        .rev()
        .find(|symbol| symbol.name == callee)
}

fn parse_signature_parameters(detail: &str) -> Option<Vec<ParameterInformation>> {
    let open = detail.find('(')?;
    let close = detail.rfind(')')?;
    if close <= open {
        return None;
    }

    let params = detail[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| ParameterInformation {
            label: ParameterLabel::Simple(item.to_string()),
            documentation: None,
        })
        .collect::<Vec<_>>();

    if params.is_empty() {
        None
    } else {
        Some(params)
    }
}

fn semantic_for_def(kind: DefKind) -> Option<(u32, u32)> {
    let token_type = match kind {
        DefKind::Class => 0,
        DefKind::Function | DefKind::Builtin => 1,
        DefKind::Method => 2,
        DefKind::Field => 3,
        DefKind::Variable => 4,
        DefKind::Parameter => 5,
    };
    Some((token_type, 1))
}

fn semantic_for_ref(kind: DefKind) -> Option<(u32, u32)> {
    let token_type = match kind {
        DefKind::Class => 0,
        DefKind::Function | DefKind::Builtin => 1,
        DefKind::Method => 2,
        DefKind::Field => 3,
        DefKind::Variable => 4,
        DefKind::Parameter => 5,
    };
    Some((token_type, 0))
}

fn semantic_tuple(
    span: Span,
    token_type: u32,
    token_modifiers_bitset: u32,
) -> (u32, u32, u32, u32, u32) {
    let line = span.line.saturating_sub(1) as u32;
    let start = span.column.saturating_sub(1) as u32;
    let mut len = span.end_column.saturating_sub(span.column) as u32;
    if len == 0 {
        len = 1;
    }
    (line, start, len, token_type, token_modifiers_bitset)
}

fn undefined_symbol_name(message: &str) -> Option<&str> {
    let prefix = "undefined symbol '";
    if !message.starts_with(prefix) {
        return None;
    }
    let rest = &message[prefix.len()..];
    let end = rest.find('\'')?;
    Some(&rest[..end])
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        state: Arc::new(RwLock::new(ServerState::default())),
    });

    Server::new(stdin, stdout, socket).serve(service).await;
}

#[cfg(test)]
mod tests {
    use super::{call_context, receiver_chain_before_dot, undefined_symbol_name};

    #[test]
    fn extracts_receiver_chain() {
        let prefix = "model.layer1.";
        assert_eq!(receiver_chain_before_dot(prefix), Some("model.layer1"));
    }

    #[test]
    fn parses_call_context() {
        let prefix = "forward(input, weights, ";
        let (callee, active) = call_context(prefix).expect("context");
        assert_eq!(callee, "forward");
        assert_eq!(active, 2);
    }

    #[test]
    fn parses_undefined_symbol() {
        assert_eq!(undefined_symbol_name("undefined symbol 'abc'"), Some("abc"));
    }
}
