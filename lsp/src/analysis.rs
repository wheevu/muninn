use muninn::typecheck::{Symbol, SymbolKind, display_ty};

pub fn markdown_for_symbol(symbol: &Symbol) -> String {
    let kind = match symbol.kind {
        SymbolKind::Global => "global",
        SymbolKind::Local => "local",
        SymbolKind::Parameter => "parameter",
        SymbolKind::Function => "function",
        SymbolKind::NativeFunction(_) => "native function",
    };

    format!(
        "```muninn\n{}\n```\n\n{} `{}`",
        symbol.detail,
        kind,
        symbol.name,
    )
}

pub fn detail_for_symbol(symbol: &Symbol) -> String {
    match symbol.kind {
        SymbolKind::NativeFunction(_) => symbol.detail.clone(),
        _ => format!("{} ({})", symbol.detail, display_ty(&symbol.ty)),
    }
}
