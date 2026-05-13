#!/usr/bin/env python

from __future__ import annotations

import argparse
import html
import json
import pathlib
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Sequence, Tuple


THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import pyscf_agent_backend as backend


BASIS_OPTIONS = (
    'sto-3g',
    '3-21g',
    '6-31g',
    '6-31g*',
    'cc-pvdz',
    'def2-svp',
)
METHOD_OPTIONS = (
    ('hf', 'Hartree-Fock'),
    ('dft', 'DFT'),
)
XC_OPTIONS = (
    'b3lyp',
    'pbe0',
    'pbe',
    'lda',
)
OUTPUT_OPTIONS = (
    ('energy', '总能'),
    ('homo_lumo', 'HOMO/LUMO'),
    ('dipole', '偶极矩'),
    ('mulliken', 'Mulliken 布居'),
)


HTML_PAGE = '''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>PySCF Agent 网页界面</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ font-family: sans-serif; margin: 0; background: #f3f6fb; color: #1f2937; }}
    h1, h2, h3 {{ margin-top: 0; }}
    textarea, select, input {{ width: 100%; box-sizing: border-box; font: inherit; }}
    textarea, select, input, button {{ border-radius: 0.75rem; border: 1px solid #cbd5e1; }}
    textarea, select, input {{ padding: 0.75rem; background: #fff; }}
    button {{ margin-top: 1rem; padding: 0.75rem 1.2rem; background: #2563eb; color: #fff; cursor: pointer; }}
    button.secondary {{ margin-left: 0.75rem; background: #475569; }}
    .page {{ max-width: 1280px; margin: 0 auto; padding: 2rem; }}
    .layout {{ display: grid; grid-template-columns: minmax(360px, 480px) 1fr; gap: 1.5rem; align-items: start; }}
    .card {{ background: #fff; border-radius: 1rem; padding: 1.25rem; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); }}
    .stack {{ display: grid; gap: 1rem; }}
    .grid-two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
    .conversation {{ min-height: 18rem; max-height: 34rem; overflow-y: auto; padding-right: 0.25rem; }}
    .message {{ margin-bottom: 0.9rem; padding: 0.85rem 1rem; border-radius: 1rem; white-space: pre-wrap; }}
    .message.user {{ background: #dbeafe; margin-left: 3rem; }}
    .message.assistant {{ background: #eef2ff; margin-right: 3rem; }}
    .message.system {{ background: #f8fafc; border: 1px dashed #cbd5e1; }}
    .message .meta {{ font-size: 0.8rem; color: #64748b; margin-bottom: 0.35rem; }}
    .checkbox-group {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.5rem; }}
    .checkbox-item {{ display: flex; align-items: center; gap: 0.5rem; padding: 0.55rem 0.7rem; background: #f8fafc; border-radius: 0.75rem; }}
    .checkbox-item input {{ width: auto; }}
    .result-list {{ margin: 0; padding-left: 1.25rem; }}
    pre {{ margin: 0; background: #0f172a; color: #e2e8f0; padding: 1rem; overflow-x: auto; white-space: pre-wrap; border-radius: 0.85rem; }}
    .hidden {{ display: none; }}
    @media (max-width: 1024px) {{
      .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>PySCF Agent 网页界面</h1>
    <p>使用结构化表单补充任务参数，通过对话面板查看交互消息，并实时预览生成的 PySCF 输入文件。</p>
    <div class="layout">
      <section class="card stack">
        <div>
          <h2>任务配置</h2>
          <p>可直接填写结构化输入，减少自然语言描述成本。</p>
        </div>
        <div>
          <label for="atom"><strong>分子结构</strong></label>
          <textarea id="atom" rows="5" placeholder="例如：O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587">{default_atom}</textarea>
        </div>
        <div class="grid-two">
          <div>
            <label for="basis"><strong>基组</strong></label>
            <select id="basis">{basis_options}</select>
          </div>
          <div>
            <label for="method"><strong>计算方法</strong></label>
            <select id="method">{method_options}</select>
          </div>
        </div>
        <div class="grid-two">
          <div>
            <label for="xc"><strong>交换关联泛函</strong></label>
            <select id="xc">{xc_options}</select>
          </div>
          <div>
            <label for="job"><strong>任务类型</strong></label>
            <select id="job">
              <option value="single_point" selected>single_point</option>
            </select>
          </div>
        </div>
        <div>
          <strong>分析输出</strong>
          <div class="checkbox-group">
            {output_options}
          </div>
        </div>
        <div>
          <label for="request"><strong>补充说明</strong></label>
          <textarea id="request" rows="4" placeholder="例如：重点关注偶极矩，若缺失信息请提醒我补充。">{default_request}</textarea>
        </div>
        <div>
          <button id="run">运行</button>
          <button id="clear" type="button" class="secondary">清空对话</button>
        </div>
      </section>
      <section class="stack">
        <div class="card">
          <h2>对话</h2>
          <div id="conversation" class="conversation"></div>
        </div>
        <div class="card stack">
          <div>
            <h2>结果摘要</h2>
            <ul id="summary" class="result-list"></ul>
          </div>
          <div>
            <h3>PySCF 输入预览</h3>
            <pre id="input-preview"></pre>
          </div>
        </div>
      </section>
    </div>
  </div>
  <script>
    function escapeHtml(value) {{
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function appendConversation(role, label, content) {{
      if (!content) return;
      const wrapper = document.createElement('div');
      wrapper.className = `message ${{role}}`;
      wrapper.innerHTML = `<div class="meta">${{escapeHtml(label)}}</div><div>${{escapeHtml(content)}}</div>`;
      const conversation = document.getElementById('conversation');
      conversation.appendChild(wrapper);
      conversation.scrollTop = conversation.scrollHeight;
    }}

    function buildRequestSummary(formData, note) {{
      const summary = [
        `结构: ${{formData.atom || '待补充'}}`,
        `基组: ${{formData.basis}}`,
        `方法: ${{formData.method.toUpperCase()}}`,
      ];
      if (formData.method === 'dft') {{
        summary.push(`泛函: ${{formData.xc}}`);
      }}
      summary.push(`分析: ${{(formData.outputs.length ? formData.outputs : ['默认']).join(', ')}}`);
      if (note) {{
        summary.push(`补充说明: ${{note}}`);
      }}
      return summary.join('\\n');
    }}

    function collectFormData() {{
      return {{
        atom: document.getElementById('atom').value.trim(),
        basis: document.getElementById('basis').value,
        method: document.getElementById('method').value,
        xc: document.getElementById('xc').value,
        job: document.getElementById('job').value,
        outputs: Array.from(document.querySelectorAll('input[name="outputs"]:checked')).map((item) => item.value),
      }};
    }}

    function syncMethodControls() {{
      const method = document.getElementById('method').value;
      const xc = document.getElementById('xc');
      xc.disabled = method !== 'dft';
    }}

    function renderSummary(payload) {{
      const summary = document.getElementById('summary');
      const items = [];
      items.push(`执行状态：${{payload.execution_status || 'unknown'}}`);
      if (payload.analysis_summary) {{
        items.push(payload.analysis_summary);
      }}
      if (payload.clarification_questions && payload.clarification_questions.length) {{
        items.push(`待补充信息：${{payload.clarification_questions.join('；')}}`);
      }}
      if (payload.structured_results && payload.structured_results.energy !== undefined) {{
        items.push(`总能：${{payload.structured_results.energy}}`);
      }}
      summary.innerHTML = items.map((item) => `<li>${{escapeHtml(item)}}</li>`).join('');
    }}

    async function runAgent() {{
      const request = document.getElementById('request').value.trim();
      const task_spec = collectFormData();
      appendConversation('user', '你', buildRequestSummary(task_spec, request));
      const response = await fetch('/api/run', {{
         method: 'POST',
         headers: {{ 'Content-Type': 'application/json' }},
         body: JSON.stringify({{ request, task_spec }})
      }});
      const payload = await response.json();
      if (!response.ok) {{
        appendConversation('system', '系统', payload.error || '请求失败');
        return;
      }}
      (payload.messages || []).filter((item) => item.role !== 'user').forEach((item) => {{
        const label = item.role === 'assistant' ? '助手' : '系统';
        appendConversation(item.role === 'assistant' ? 'assistant' : 'system', label, item.content);
      }});
      document.getElementById('input-preview').textContent = payload.generated_input || '';
      renderSummary(payload);
    }}

    document.getElementById('method').addEventListener('change', syncMethodControls);
    document.getElementById('run').addEventListener('click', runAgent);
    document.getElementById('clear').addEventListener('click', () => {{
      document.getElementById('conversation').innerHTML = '';
      document.getElementById('summary').innerHTML = '';
      document.getElementById('input-preview').textContent = '';
    }});
    syncMethodControls();
  </script>
</body>
</html>
'''


def _build_options(options: Sequence[Any], selected: str) -> str:
    rendered = []
    for option in options:
        if isinstance(option, tuple):
            value, label = option
        else:
            value = label = option
        attrs = ' selected' if value == selected else ''
        rendered.append(
            '<option value="{value}"{attrs}>{label}</option>'.format(
                value=html.escape(str(value)),
                attrs=attrs,
                label=html.escape(str(label)),
            )
        )
    return '\n'.join(rendered)


def _build_output_options(selected_outputs: Sequence[str]) -> str:
    rendered = []
    for value, label in OUTPUT_OPTIONS:
        checked = ' checked' if value in selected_outputs else ''
        rendered.append(
            '<label class="checkbox-item"><input type="checkbox" name="outputs" value="{value}"{checked}>{label}</label>'.format(
                value=html.escape(value),
                checked=checked,
                label=html.escape(label),
            )
        )
    return '\n'.join(rendered)


def build_agent_request(payload: Dict[str, Any]) -> str:
    request_text = payload.get('request')
    task_spec_payload = payload.get('task_spec')
    task_spec = task_spec_payload if isinstance(task_spec_payload, dict) else {}
    request_note = request_text.strip() if isinstance(request_text, str) else ''
    structured_request = {}
    for key in ('atom', 'basis', 'method', 'xc', 'job'):
        value = task_spec.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value:
            structured_request[key] = value
    outputs = task_spec.get('outputs')
    if isinstance(outputs, list):
        normalized_outputs = []
        for item in outputs:
            normalized_item = str(item).strip()
            if normalized_item:
                normalized_outputs.append(normalized_item)
        if normalized_outputs:
            structured_request['outputs'] = normalized_outputs
    if request_note:
        structured_request['request'] = request_note
    if structured_request:
        return json.dumps(structured_request, ensure_ascii=False)
    return request_note


def build_index_html() -> str:
    try:
        example_request = json.loads(backend.example_request())
    except (TypeError, json.JSONDecodeError):
        backend.LOGGER.warning('Failed to parse example_request JSON, using empty defaults', exc_info=True)
        example_request = {}
    return HTML_PAGE.format(
        default_atom=html.escape(example_request.get('atom', '')),
        default_request='',
        basis_options=_build_options(BASIS_OPTIONS, example_request.get('basis', '6-31g')),
        method_options=_build_options(METHOD_OPTIONS, example_request.get('method', 'dft')),
        xc_options=_build_options(XC_OPTIONS, example_request.get('xc', 'b3lyp')),
        output_options=_build_output_options(example_request.get('outputs', [])),
    )


def handle_api_request(request_body: bytes) -> Tuple[HTTPStatus, Dict[str, str], bytes]:
    try:
        payload = json.loads(request_body.decode('utf-8') or '{}')
    except json.JSONDecodeError:
        response = {'error': 'Invalid JSON body'}
        return HTTPStatus.BAD_REQUEST, {'Content-Type': 'application/json; charset=utf-8'}, (
            json.dumps(response, ensure_ascii=False).encode('utf-8')
        )

    request_text = build_agent_request(payload)
    if not request_text:
        response = {'error': 'Field "request" or "task_spec" is required'}
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
