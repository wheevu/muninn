mod analysis;
mod convert;
mod state;

use std::sync::Arc;

use analysis::{detail_for_symbol, markdown_for_symbol};
use convert::{errors_to_diagnostics, span_to_range};
use muninn::analyze_document;
use state::{DocumentState, ServerState, apply_content_changes, is_stale_version};
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

#[derive(Clone)]
struct Backend {
    client: Client,
    state: Arc<RwLock<ServerState>>,
}

impl Backend {
    async fn update_document(&self, uri: Url, version: i32, source: String) {
        let analysis = analyze_document(&source);
        let diagnostics = errors_to_diagnostics(&source, &muninn::source::compute_line_starts(&source), &uri, &analysis.diagnostics);
        let document = Arc::new(DocumentState::new(version, source, analysis));
        {
            let mut state = self.state.write().await;
            if let Some(existing) = state.docs.get(&uri)
                && is_stale_version(existing.version, version)
            {
                return;
            }
            state.docs.insert(uri.clone(), document);
        }
        self.client.publish_diagnostics(uri, diagnostics, None).await;
    }

    async fn document(&self, uri: &Url) -> Option<Arc<DocumentState>> {
        self.state.read().await.docs.get(uri).cloned()
    }

    fn schedule_document_update(&self, uri: Url, version: i32, source: String) {
        let backend = self.clone();
        tokio::spawn(async move {
            backend.update_document(uri, version, source).await;
        });
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::INCREMENTAL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                ..ServerCapabilities::default()
            },
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
        self.schedule_document_update(uri, version, source);
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;
        let current = self.document(&uri).await;
        let source = if let Some(current) = current {
            apply_content_changes(&current.source, &params.content_changes)
        } else {
            params.content_changes.first().map(|change| change.text.clone())
        };
        let Some(source) = source else {
            self.client
                .log_message(MessageType::WARNING, "failed to apply text changes")
                .await;
            return;
        };
        self.schedule_document_update(uri, version, source);
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        self.state.write().await.docs.remove(&params.text_document.uri);
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
        let Some(offset) = doc.offset_at(position.line, position.character) else {
            return Ok(None);
        };
        let Some(symbol) = doc.analysis.definition_at_offset(offset) else {
            return Ok(None);
        };
        Ok(Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!("{}\n\n{}", markdown_for_symbol(symbol), detail_for_symbol(symbol)),
            }),
            range: Some(span_to_range(&doc.source, &doc.line_starts, symbol.span)),
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
        let Some(offset) = doc.offset_at(position.line, position.character) else {
            return Ok(None);
        };
        let Some(symbol) = doc.analysis.definition_at_offset(offset) else {
            return Ok(None);
        };
        Ok(Some(GotoDefinitionResponse::Scalar(Location {
            uri,
            range: span_to_range(&doc.source, &doc.line_starts, symbol.span),
        })))
    }
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
    use muninn::{analyze_document, source::offset_to_utf16_position};

    use crate::analysis::markdown_for_symbol;
    use crate::state::DocumentState;

    #[test]
    fn resolves_definition_for_function_call_site() {
        let source = "fn add(a: Int, b: Int) -> Int { return a + b; }\nlet value: Int = add(1, 2);\n";
        let analysis = analyze_document(source);
        let doc = DocumentState::new(1, source.to_string(), analysis);

        let call_offset = source.find("add(1, 2)").expect("call offset");
        let (line, character) =
            offset_to_utf16_position(source, &doc.line_starts, call_offset);
        let resolved_offset = doc.offset_at(line, character).expect("resolved offset");
        let semantics = doc.analysis.semantics.as_ref().expect("semantics");
        let symbol = semantics
            .definition_at_offset(resolved_offset)
            .expect("definition");

        assert_eq!(symbol.name, "add");
        assert!(markdown_for_symbol(symbol).contains("fn add"));
    }

    #[test]
    fn assignment_reference_span_is_identifier_only() {
        let source = "let mut x: Int = 1;\nx = 2;\nx;\n";
        let analysis = analyze_document(source);
        let doc = DocumentState::new(1, source.to_string(), analysis);
        let semantics = doc.analysis.semantics.as_ref().expect("semantics");

        let assign_name_offset = source.find("x = 2").expect("assign name");
        let symbol = semantics
            .definition_at_offset(assign_name_offset)
            .expect("assign definition");
        assert_eq!(symbol.name, "x");

        let literal_offset = source.find("2;").expect("literal");
        assert!(semantics.definition_at_offset(literal_offset).is_none());
    }

    #[test]
    fn utf16_round_trip_offsets_for_symbol_lookup() {
        let source = "let bird: String = \"🐦\";\nprint(bird);\n";
        let analysis = analyze_document(source);
        let doc = DocumentState::new(1, source.to_string(), analysis);
        let symbol_offset = source.find("bird);\n").expect("bird call");
        let (line, character) =
            offset_to_utf16_position(source, &doc.line_starts, symbol_offset);
        let round_trip = doc.offset_at(line, character).expect("offset");
        assert_eq!(round_trip, symbol_offset);
        let semantics = doc.analysis.semantics.as_ref().expect("semantics");
        assert!(semantics.definition_at_offset(round_trip).is_some());
    }

    #[test]
    fn definition_lookup_survives_unrelated_diagnostics() {
        let source = "let value: Int = 1;\nlet bad: Int = true;\nvalue;\n";
        let analysis = analyze_document(source);
        assert!(!analysis.diagnostics.is_empty());
        let doc = DocumentState::new(2, source.to_string(), analysis);

        let use_offset = source.rfind("value;").expect("value use");
        let symbol = doc
            .analysis
            .definition_at_offset(use_offset)
            .expect("definition");
        assert_eq!(symbol.name, "value");
    }
}
