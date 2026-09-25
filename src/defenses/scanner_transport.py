"""Opt-in backend compatibility for native scanner chat requests.

Qwen's retained profile needs enable_thinking=false. Cisco strict-schema
compatibility removes only uniqueItems on evidence_ids array properties.
No other schema constraint or message content is changed.
"""
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def adapt_request(payload, *, disable_thinking=False, evidence_ids_compat=False):
    result = deepcopy(payload)
    if disable_thinking:
        options = result.setdefault('chat_template_kwargs', {})
        if not isinstance(options, dict):
            raise ValueError('chat_template_kwargs must be an object')
        options['enable_thinking'] = False
    if evidence_ids_compat:
        schema = result.get('response_format', {}).get('json_schema', {}).get('schema')
        def visit(node):
            if isinstance(node, dict):
                properties = node.get('properties', {})
                field = properties.get('evidence_ids') if isinstance(properties, dict) else None
                if isinstance(field, dict) and field.get('type') == 'array':
                    field.pop('uniqueItems', None)
                for value in node.values(): visit(value)
            elif isinstance(node, list):
                for value in node: visit(value)
        visit(schema)
    return result


@contextmanager
def compatibility_endpoint(upstream, *, disable_thinking=False, evidence_ids_compat=False, timeout=120):
    # Local transport only; the configured upstream performs model inference.
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_error(404); return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict) or payload.get('stream'):
                    raise ValueError('Only non-streaming JSON chat requests are supported')
                body = json.dumps(adapt_request(payload, disable_thinking=disable_thinking,
                                                evidence_ids_compat=evidence_ids_compat)).encode()
                headers = {'Content-Type': 'application/json'}
                if self.headers.get('Authorization'):
                    headers['Authorization'] = self.headers['Authorization']
                request = Request(upstream.rstrip('/') + '/chat/completions', data=body, headers=headers)
                try:
                    with urlopen(request, timeout=timeout) as response:
                        status, data = response.status, response.read()
                except HTTPError as exc:
                    status, data = exc.code, exc.read()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers(); self.wfile.write(data)
            except Exception:
                self.send_error(502, 'Scanner backend request failed')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1'
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
