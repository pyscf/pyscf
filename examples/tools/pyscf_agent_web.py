#!/usr/bin/env python

from __future__ import annotations

import argparse
import html
import json
import pathlib
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Sequence, Tuple


THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import pyscf_agent_backend as backend


HTML_PAGE = '''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>PySCF Agent 网页界面</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; max-width: 960px; }}
    textarea {{ width: 100%; min-height: 12rem; }}
    button {{ margin-top: 1rem; padding: 0.6rem 1.2rem; }}
    pre {{ background: #f6f8fa; padding: 1rem; overflow-x: auto; white-space: pre-wrap; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  </style>
</head>
<body>
  <h1>PySCF Agent 网页界面</h1>
  <p>统一后端、结构化消息和结构化日志演示。</p>
  <textarea id="request">{default_request}</textarea>
  <br>
  <button id="run">运行</button>
  <div class="row">
    <div>
      <h2>最终报告</h2>
      <pre id="report"></pre>
    </div>
    <div>
      <h2>结构化日志</h2>
      <pre id="logs"></pre>
    </div>
  </div>
  <script>
    async function runAgent() {{
      const request = document.getElementById('request').value;
      const response = await fetch('/api/run', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ request }})
      }});
      const payload = await response.json();
      document.getElementById('report').textContent = JSON.stringify(payload, null, 2);
      document.getElementById('logs').textContent = JSON.stringify(payload.logs || [], null, 2);
    }}
    document.getElementById('run').addEventListener('click', runAgent);
  </script>
</body>
</html>
'''


def build_index_html() -> str:
    return HTML_PAGE.format(default_request=html.escape(backend.example_request()))


def handle_api_request(request_body: bytes) -> Tuple[HTTPStatus, Dict[str, str], bytes]:
    try:
        payload = json.loads(request_body.decode('utf-8') or '{}')
    except json.JSONDecodeError:
        response = {'error': 'Invalid JSON body'}
        return HTTPStatus.BAD_REQUEST, {'Content-Type': 'application/json; charset=utf-8'}, (
            json.dumps(response, ensure_ascii=False).encode('utf-8')
        )

    request_text = payload.get('request')
    if not request_text:
        response = {'error': 'Field "request" is required'}
        return HTTPStatus.BAD_REQUEST, {'Content-Type': 'application/json; charset=utf-8'}, (
            json.dumps(response, ensure_ascii=False).encode('utf-8')
        )

    report = backend.execute_request(request_text, channel='web')['final_report']
    return HTTPStatus.OK, {'Content-Type': 'application/json; charset=utf-8'}, (
        json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8')
    )


class AgentWebHandler(BaseHTTPRequestHandler):
    def _write_response(self, status: HTTPStatus, headers: Dict[str, str], body: bytes) -> None:
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != '/':
            self._write_response(
                HTTPStatus.NOT_FOUND,
                {'Content-Type': 'text/plain; charset=utf-8'},
                b'Not Found',
            )
            return
        body = build_index_html().encode('utf-8')
        self._write_response(HTTPStatus.OK, {'Content-Type': 'text/html; charset=utf-8'}, body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != '/api/run':
            self._write_response(
                HTTPStatus.NOT_FOUND,
                {'Content-Type': 'text/plain; charset=utf-8'},
                b'Not Found',
            )
            return
        content_length = int(self.headers.get('Content-Length', '0'))
        request_body = self.rfile.read(content_length)
        status, headers, body = handle_api_request(request_body)
        self._write_response(status, headers, body)

    def log_message(self, format: str, *args: object) -> None:
        backend.LOGGER.info(json.dumps({
            'level': 'info',
            'event': 'web.http_access',
            'details': {
                'client': self.address_string(),
                'path': self.path,
                'message': format % args,
            },
                'timestamp': backend.utc_timestamp(),
        }, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Serve the PySCF agent web frontend')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    return parser


def serve(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), AgentWebHandler)
    print('Serving on http://{0}:{1}'.format(args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(serve())
