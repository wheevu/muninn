use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const MESSAGE_TIMEOUT: Duration = Duration::from_secs(5);
const EXIT_TIMEOUT: Duration = Duration::from_secs(5);

struct LspProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    messages: Receiver<Value>,
    next_id: u64,
    shutdown_sent: bool,
    exited: bool,
}

impl LspProcess {
    fn start() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_muninn-lsp"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start muninn-lsp");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let (sender, messages) = mpsc::channel();
        thread::spawn(move || read_stdout_messages(stdout, sender));
        Self {
            child,
            stdin: Some(stdin),
            messages,
            next_id: 1,
            shutdown_sent: false,
            exited: false,
        }
    }

    fn request(&mut self, method: &str, params: Value) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.send(json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }));
        id
    }

    fn notify(&mut self, method: &str, params: Value) {
        self.send(json!({ "jsonrpc": "2.0", "method": method, "params": params }));
    }

    fn send(&mut self, message: Value) {
        let body = serde_json::to_vec(&message).expect("json");
        let stdin = self.stdin.as_mut().expect("stdin open");
        write!(stdin, "Content-Length: {}\r\n\r\n", body.len()).expect("header");
        stdin.write_all(&body).expect("body");
        stdin.flush().expect("flush");
    }

    fn read_message(&mut self) -> Value {
        self.messages
            .recv_timeout(MESSAGE_TIMEOUT)
            .expect("timed out waiting for LSP message")
    }

    fn shutdown(&mut self) {
        if self.shutdown_sent {
            return;
        }
        self.shutdown_sent = true;
        let shutdown_id = self.request("shutdown", json!(null));
        loop {
            let message = self.read_message();
            if message.get("id") == Some(&json!(shutdown_id)) {
                break;
            }
        }
        self.notify("exit", json!(null));
        drop(self.stdin.take());
        let deadline = Instant::now() + EXIT_TIMEOUT;
        while Instant::now() < deadline {
            if self.child.try_wait().expect("poll lsp exit").is_some() {
                self.exited = true;
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.exited = true;
        panic!("muninn-lsp did not exit after shutdown");
    }
}

impl Drop for LspProcess {
    fn drop(&mut self) {
        if !self.exited {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

fn read_stdout_messages(mut stdout: impl Read, sender: mpsc::Sender<Value>) {
    while let Ok(message) = read_message(&mut stdout) {
        if sender.send(message).is_err() {
            return;
        }
    }
}

fn read_message(stdout: &mut impl Read) -> std::io::Result<Value> {
    let mut header = Vec::new();
    let mut byte = [0u8; 1];
    while !header.ends_with(b"\r\n\r\n") {
        stdout.read_exact(&mut byte)?;
        header.push(byte[0]);
    }
    let header = String::from_utf8(header)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let length = header
        .lines()
        .find_map(|line| line.strip_prefix("Content-Length: "))
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "content length"))?;
    let mut body = vec![0u8; length];
    stdout.read_exact(&mut body)?;
    serde_json::from_slice(&body)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}

#[test]
fn lsp_publishes_diagnostics_with_utf16_ranges_over_json_rpc() {
    let mut lsp = LspProcess::start();
    let initialize_id = lsp.request(
        "initialize",
        json!({
            "processId": null,
            "rootUri": null,
            "capabilities": {}
        }),
    );

    loop {
        let message = lsp.read_message();
        if message.get("id") == Some(&json!(initialize_id)) {
            break;
        }
    }

    lsp.notify("initialized", json!({}));
    lsp.notify(
        "textDocument/didOpen",
        json!({
            "textDocument": {
                "uri": "file:///tmp/muninn-lsp-utf16.mun",
                "languageId": "muninn",
                "version": 1,
                "text": "let text: String = \"🐦\"; let bad: Int = true;\n"
            }
        }),
    );

    loop {
        let message = lsp.read_message();
        if message.get("method") != Some(&json!("textDocument/publishDiagnostics")) {
            continue;
        }
        let diagnostic = &message["params"]["diagnostics"][0];
        assert!(diagnostic["message"].as_str().unwrap_or("").contains("Int"));
        assert_eq!(diagnostic["range"]["start"]["line"], 0);
        assert_eq!(diagnostic["range"]["start"]["character"], 40);
        assert_eq!(diagnostic["range"]["end"]["line"], 0);
        assert_eq!(diagnostic["range"]["end"]["character"], 44);
        break;
    }

    lsp.shutdown();
}
