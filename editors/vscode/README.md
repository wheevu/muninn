# Muninn VS Code Extension

This extension provides thin VS Code integration for Muninn (`.mun`) files.

## Features

- File association for `.mun`
- Syntax highlighting
- Language Server support for:
  - diagnostics
  - hover
  - go to definition

## Requirements

- VS Code 1.85+
- `muninn-lsp` available either:
  - via `muninn.serverPath`
  - at `target/debug/muninn-lsp` in the opened workspace
  - on your `PATH`

## Development

From repository root:

```bash
cargo build -p muninn-lsp
cd editors/vscode
npm install
npm run compile
```

Open `editors/vscode` in VS Code and press `F5` to launch the Extension Development Host.
