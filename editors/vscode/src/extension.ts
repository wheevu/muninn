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
  const server = resolveServerCommand();
  const serverOptions: ServerOptions = {
    run: {
      command: server.command,
      args: server.args,
      options: server.cwd ? { cwd: server.cwd } : undefined,
      transport: TransportKind.stdio,
    },
    debug: {
      command: server.command,
      args: server.args,
      options: server.cwd ? { cwd: server.cwd } : undefined,
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

function resolveServerCommand(): { command: string; args: string[]; cwd?: string } {
  const configured = vscode.workspace
    .getConfiguration("muninn")
    .get<string>("serverPath", "")
    .trim();
  if (configured.length > 0) {
    return { command: configured, args: [] };
  }

  const executable = process.platform === "win32" ? "muninn-lsp.exe" : "muninn-lsp";
  const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (workspaceRoot) {
    const localBuild = path.join(workspaceRoot, "target", "debug", executable);
    if (fs.existsSync(localBuild)) {
      return { command: localBuild, args: [] };
    }

    const rootCargo = path.join(workspaceRoot, "Cargo.toml");
    const lspCargo = path.join(workspaceRoot, "lsp", "Cargo.toml");
    if (fs.existsSync(rootCargo) && fs.existsSync(lspCargo)) {
      return {
        command: "cargo",
        args: ["run", "--quiet", "-p", "muninn-lsp"],
        cwd: workspaceRoot,
      };
    }
  }

  return { command: executable, args: [] };
}
