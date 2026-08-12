#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve an OD defect gallery on the first available port."""

from __future__ import annotations

import argparse
import json
import socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--endpoint-file", required=True)
    parser.add_argument("--url-path", default="/synthetic_gallery/#gallery")
    parser.add_argument("--port-start", type=int, default=8080)
    parser.add_argument("--port-end", type=int, default=8200)
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    if not (directory / "synthetic_gallery" / "index.html").is_file():
        raise FileNotFoundError(directory / "synthetic_gallery" / "index.html")
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = None
    for port in range(args.port_start, args.port_end):
        try:
            server = ThreadingHTTPServer(("0.0.0.0", port), handler)
            break
        except OSError as error:
            if error.errno != 98:
                raise
    if server is None:
        raise RuntimeError(f"no free port in [{args.port_start}, {args.port_end})")

    node_ip = socket.gethostbyname(socket.gethostname())
    url = f"http://{node_ip}:{server.server_port}{args.url_path}"
    endpoint = Path(args.endpoint_file)
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    endpoint.write_text(
        json.dumps(
            {"state": "RUNNING", "url": url, "port": server.server_port, "node_ip": node_ip},
            indent=2,
        )
        + "\n"
    )
    print(f"GALLERY_URL={url}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
