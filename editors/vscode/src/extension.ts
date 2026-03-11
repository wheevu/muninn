import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const command = resolveServerPath(context);
  const serverOptions: ServerOptions = {
    run: {
      command,
      transport: TransportKind.stdio,
    },
    debug: {
      command,
      transport: TransportKind.stdio,
    },
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "muninn" }],
    outputChannel: vscode.window.createOutputChannel("Muninn Language Server"),
  };

  client = new LanguageClient(
    "muninn-lsp",
    "Muninn Language Server",
    serverOptions,
    clientOptions,
  );

  await client.start();
  context.subscriptions.push({
    dispose: () => {
      void client?.stop();
    },
  });
}

export async function deactivate(): Promise<void> {
  if (!client) {
    return;
  }
  await client.stop();
  client = undefined;
}

function resolveServerPath(context: vscode.ExtensionContext): string {
  const configured = vscode.workspace
    .getConfiguration("muninn")
    .get<string>("serverPath", "")
    .trim();
  if (configured.length > 0) {
    return configured;
  }

  const executable = process.platform === "win32" ? "muninn-lsp.exe" : "muninn-lsp";

  const bundled = context.asAbsolutePath(path.join("bin", executable));
  if (fs.existsSync(bundled)) {
    return bundled;
  }

  const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (workspaceRoot) {
    const localBuild = path.join(workspaceRoot, "target", "debug", executable);
    if (fs.existsSync(localBuild)) {
      return localBuild;
    }
  }

  return "muninn-lsp";
}
