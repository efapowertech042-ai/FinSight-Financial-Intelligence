import "dotenv/config";
import express from "express";
import http from "http";
import path from "path";
import fs from "fs";
import { spawn, type ChildProcess } from "child_process";
import { fileURLToPath } from "url";
import { GoogleGenAI } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
// Use isolated port 8765 so FastAPI never collides with container control-plane ports (8000/8080)
const PY_PORT = Number(process.env.PY_PORT || 8765);
const PY_HOST = "127.0.0.1";

let pyProcess: ChildProcess | null = null;
let pyReady = false;
let pyProbeInFlight = false;

const PY_LIBS_DIR = "/tmp/.pylibs";

function buildPythonPath(): string {
  const candidates = [
    PY_LIBS_DIR,
    path.join(__dirname, ".pylibs"),
    process.env.PYTHONPATH || "",
  ].filter(Boolean);
  return candidates.join(path.delimiter);
}

function resolvePythonExecutable(): string {
  const venvPy = path.join(__dirname, ".venv", "bin", "python3");
  if (fs.existsSync(venvPy)) {
    return venvPy;
  }
  return "python3";
}

function startPythonBackend(): void {
  if (pyProcess) return;
  fs.mkdirSync(PY_LIBS_DIR, { recursive: true });
  const pyBin = resolvePythonExecutable();
  const pyPath = buildPythonPath();
  console.log(`[Python FastAPI] Starting backend on ${PY_HOST}:${PY_PORT} using ${pyBin}...`);

  const launchScript = `
import importlib, os, subprocess, sys

libs_dir = "${PY_LIBS_DIR}"
os.makedirs(libs_dir, exist_ok=True)
if libs_dir not in sys.path:
    sys.path.insert(0, libs_dir)
sys.path_importer_cache.clear()
importlib.invalidate_caches()

def _verify_deps():
    import fastapi, uvicorn, pandas, numpy, openpyxl  # noqa: F401

try:
    _verify_deps()
except ImportError:
    print("[FastAPI] Bootstrapping Python ML dependencies into " + libs_dir + "...", flush=True)
    env = dict(os.environ, PYTHONPATH=libs_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))
    pip_check = subprocess.run([sys.executable, "-m", "pip", "--version"], env=env, capture_output=True)
    if pip_check.returncode != 0:
        import urllib.request
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", "/tmp/get-pip.py")
        subprocess.check_call([sys.executable, "/tmp/get-pip.py", "--target", libs_dir, "--no-warn-script-location"], env=env)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "--target", libs_dir, "-r", "backend/requirements.txt"],
        env=env,
    )
    sys.path_importer_cache.clear()
    importlib.invalidate_caches()
    os.execve(
        sys.executable,
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "${PY_HOST}", "--port", "${PY_PORT}", "--log-level", "info"],
        env,
    )

import uvicorn
uvicorn.run("backend.main:app", host="${PY_HOST}", port=${PY_PORT}, log_level="info")
`;

  pyProcess = spawn(pyBin, ["-c", launchScript], {
    cwd: __dirname,
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONPATH: pyPath },
  });

  pyProcess.stdout?.on("data", (chunk) => {
    process.stdout.write(`[FastAPI] ${chunk}`);
  });
  pyProcess.stderr?.on("data", (chunk) => {
    process.stderr.write(`[FastAPI] ${chunk}`);
  });
  pyProcess.on("exit", (code, signal) => {
    console.error(`[FastAPI] exited with code=${code}, signal=${signal}`);
    pyProcess = null;
    pyReady = false;
    pyProbeInFlight = false;
  });

  void monitorPythonReady();
}

async function probePythonHealth(): Promise<boolean> {
  try {
    const res = await fetch(`http://${PY_HOST}:${PY_PORT}/api/py/health`);
    if (res.ok) {
      if (!pyReady) {
        console.log(`[Python FastAPI] Ready on ${PY_HOST}:${PY_PORT}`);
      }
      pyReady = true;
      return true;
    }
  } catch {
    // uvicorn still starting or installing dependencies
  }
  return false;
}

async function monitorPythonReady(): Promise<void> {
  if (pyProbeInFlight) return;
  pyProbeInFlight = true;
  try {
    while (pyProcess && !pyReady) {
      if (await probePythonHealth()) {
        return;
      }
      await new Promise((r) => setTimeout(r, 500));
    }
  } finally {
    pyProbeInFlight = false;
  }
}

async function ensurePythonReady(maxWaitMs = 4000): Promise<boolean> {
  if (pyReady) return true;
  if (!pyProcess) {
    startPythonBackend();
  }
  const deadline = Date.now() + maxWaitMs;
  while (Date.now() < deadline) {
    if (pyReady || (await probePythonHealth())) {
      return true;
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}

async function proxyToFastAPI(req: express.Request, res: express.Response): Promise<void> {
  const ready = await ensurePythonReady(4000);
  if (!ready) {
    res.status(503).json({
      detail: "Python analytical backend is warming up. Retrying automatically...",
      warming_up: true,
    });
    return;
  }

  const targetPath = req.originalUrl;
  const headers: Record<string, string | string[] | undefined> = { ...req.headers };
  delete headers.host;

  const proxyReq = http.request(
    {
      hostname: PY_HOST,
      port: PY_PORT,
      path: targetPath,
      method: req.method,
      headers,
    },
    (proxyRes) => {
      res.status(proxyRes.statusCode || 500);
      for (const [k, v] of Object.entries(proxyRes.headers)) {
        if (v !== undefined) {
          res.setHeader(k, v);
        }
      }
      proxyRes.pipe(res);
    }
  );

  proxyReq.on("error", (err) => {
    console.error("[Proxy Error]", err.message);
    if (!res.headersSent) {
      res.status(502).json({
        detail: "Python analytical backend is unavailable.",
        error: err.message,
      });
    }
  });

  req.pipe(proxyReq);
}

async function createServer(): Promise<void> {
  // Start Python FastAPI asynchronously so port 3000 binds immediately
  startPythonBackend();

  const app = express();

  // Proxy all /api/py/* routes directly (streams multipart uploads & CSV downloads intact)
  app.use("/api/py", (req, res) => {
    void proxyToFastAPI(req, res);
  });

  // Graphical RAG endpoint combining Python retrieval + optional Gemini synthesis
  app.use("/api/rag/ask", express.json(), async (req, res) => {
    try {
      const ready = await ensurePythonReady(4000);
      if (!ready) {
        return res.status(503).json({
          error: "Python analytical backend is warming up. Please retry in a moment.",
          warming_up: true,
        });
      }

      const { question = "", lens = "profit_intelligence", selected_month = null } = req.body || {};

      const pyRes = await fetch(`http://${PY_HOST}:${PY_PORT}/api/py/rag/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, lens, selected_month }),
      });
      if (!pyRes.ok) {
        const errText = await pyRes.text();
        return res.status(pyRes.status).json({ error: errText });
      }
      const retrieved = await pyRes.json();

      // Attempt grounded synthesis via Gemini API ONLY if a real key is configured
      const apiKey = process.env.GEMINI_API_KEY;
      let llmSynthesis: string | null = null;
      let llmGrounded = false;
      let synthesisMode = "Deterministic Retrieval-Grounded Explanation";

      if (apiKey && apiKey !== "MY_GEMINI_API_KEY" && apiKey.trim().length > 10) {
        try {
          const ai = new GoogleGenAI({
            apiKey,
            httpOptions: {
              headers: { "User-Agent": "aistudio-build" },
            },
          });

          const groundedPrompt = [
            "You are a senior financial intelligence analyst.",
            "Answer the user's analytical question strictly using ONLY the retrieved evidence, ledger records, and retrieval process trace below.",
            "Do NOT invent any numbers, months, or findings not present in the retrieved evidence.",
            "Provide 3-4 concise sentences explaining both the exact quantitative figures retrieved from the dataset and the decision implication.",
            `User Question: ${retrieved.question}`,
            `Data Provenance: ${JSON.stringify(retrieved.data_provenance)}`,
            `Matched Dataset Ledger Rows: ${JSON.stringify(retrieved.matched_records)}`,
            `Retrieval Process Trace: ${JSON.stringify(retrieved.retrieval_process)}`,
            `Retrieved Evidence: ${JSON.stringify(retrieved.evidence)}`,
            `Baseline Summary: ${retrieved.analytical_explanation}`,
            `Business Implications: ${retrieved.business_implications}`,
          ].join("\n\n");

          const response = await ai.models.generateContent({
            model: "gemini-4-hydrogen",
            contents: groundedPrompt,
          });
          if (response.text && response.text.trim()) {
            llmSynthesis = response.text.trim();
            llmGrounded = true;
            synthesisMode = "Retrieval-Grounded Gemini Synthesis";
          }
        } catch (geminiErr: unknown) {
          const msg = geminiErr instanceof Error ? geminiErr.message : String(geminiErr);
          console.warn("[RAG Gemini Fallback]", msg);
        }
      }

      return res.json({
        ...retrieved,
        synthesis_mode: synthesisMode,
        llm_grounded: llmGrounded,
        llm_narrative: llmSynthesis,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return res.status(500).json({ error: msg });
    }
  });

  // Guarantee unmatched /api/* requests return JSON instead of falling through to index.html
  app.use("/api", (_req, res) => {
    res.status(404).json({ error: "API route not found." });
  });

  if (process.env.NODE_ENV === "production") {
    const distPath = path.join(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  } else {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        // Explicitly disable HMR & WebSocket server in iframe/proxied dev environments
        hmr: false,
        ws: false,
        watch: null,
      },
      appType: "spa",
    });
    // Serve a clean no-WebSocket @vite/client shim so CSS injection works without HMR WS errors
    app.get("/@vite/client", (_req, res) => {
      res.type("application/javascript").send(`
const sheetsMap = new Map();
export function updateStyle(id, content) {
  let style = sheetsMap.get(id);
  if (!style) {
    style = document.createElement("style");
    style.setAttribute("type", "text/css");
    style.setAttribute("data-vite-dev-id", id);
    style.textContent = content;
    document.head.appendChild(style);
    sheetsMap.set(id, style);
  } else {
    style.textContent = content;
  }
}
export function removeStyle(id) {
  const style = sheetsMap.get(id);
  if (style) {
    document.head.removeChild(style);
    sheetsMap.delete(id);
  }
}
export function createHotContext() {
  return {
    data: {},
    accept() {},
    acceptExports() {},
    dispose() {},
    prune() {},
    decline() {},
    invalidate() {},
    on() {},
    off() {},
    send() {},
  };
}
export function injectQuery(url) { return url; }
export class ErrorOverlay {}
`);
    });
    app.use(vite.middlewares);
  }

  const server = app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Server] FinSight platform listening on http://0.0.0.0:${PORT}`);
  });

  const shutdown = () => {
    if (pyProcess) {
      pyProcess.kill();
    }
    server.close(() => process.exit(0));
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

createServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
