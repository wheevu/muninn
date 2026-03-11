# Muninn VS Code Extension

This extension provides language support for Muninn (`.mun`) files.

## Features

- File association for `.mun`
- Syntax highlighting (TextMate grammar)
- Language configuration (comments, brackets, indentation)
- Language Server support:
  - diagnostics
  - hover
  - go to definition
  - references
  - rename symbol
  - document symbols
  - workspace symbols (open documents)
  - completion (keywords, symbols, members, types)
  - signature help
  - semantic tokens
  - quick-fix code actions for common parser/typechecker errors

## Requirements

- VS Code 1.85+
- `muninn-lsp` binary available either:
  - in extension setting `muninn.serverPath`, or
  - at `target/debug/muninn-lsp` in the opened workspace, or
  - on your `PATH`

## Development

From repository root:

```bash
cargo build -p muninn-lsp
cd editors/vscode
npm install
npm run compile
```

Open `editors/vscode` in VS Code and press `F5` to launch Extension Development Host.

## Configuration

- `muninn.serverPath`: Optional absolute path to the language server binary.
