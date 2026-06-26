#!/usr/bin/env python3
"""Copy a single-arch Docker/OCI image manifest between registries.

This is intended for large images where `docker pull && docker push` would fill
the local Docker data root. It streams blobs from source registry to destination
registry and only reads credentials from ~/.docker/config.json.

Known target quirk:
  micr.cloud.mioffice.cn is Harbor. Upload requests must use a Harbor token and
  must not carry Harbor web cookies, otherwise POST uploads can fail with
  "CSRF token invalid".
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.index.v1+json",
    ]
)

DEFAULT_MICR_REALM = "https://micr-internal.cloud.mioffice.cn/service/token"


@dataclass(frozen=True)
class ImageRef:
    host: str
    repo: str
    ref: str

    @classmethod
    def parse(cls, value: str) -> "ImageRef":
        value = value.removeprefix("docker://")
        if "/" not in value:
            raise ValueError(f"invalid image ref: {value}")
        host, rest = value.split("/", 1)
        if "@" in rest:
            repo, ref = rest.rsplit("@", 1)
        else:
            repo, ref = rest.rsplit(":", 1)
        return cls(host=host, repo=repo, ref=ref)

    def __str__(self) -> str:
        sep = "@" if self.ref.startswith("sha256:") else ":"
        return f"{self.host}/{self.repo}{sep}{self.ref}"


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=False,
    )
    session.mount(
        "https://",
        HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8),
    )
    return session


def parse_www_authenticate(value: Optional[str]) -> Dict[str, str]:
    if not value or not value.lower().startswith("bearer "):
        return {}
    return dict(re.findall(r'(\w+)="([^"]*)"', value))


def load_docker_auth(host: str, authfile: str) -> Optional[Tuple[str, str]]:
    with open(os.path.expanduser(authfile), "r", encoding="utf-8") as f:
        data = json.load(f)
    auths = data.get("auths", {})
    for key in (host, f"https://{host}", f"http://{host}"):
        item = auths.get(key)
        if not item:
            continue
        if "auth" in item:
            raw = base64.b64decode(item["auth"]).decode("utf-8")
            return tuple(raw.split(":", 1))  # type: ignore[return-value]
        if "username" in item and "password" in item:
            return item["username"], item["password"]
    return None


class RegistryClient:
    def __init__(
        self,
        image: ImageRef,
        authfile: str,
        *,
        micr_realm: str = DEFAULT_MICR_REALM,
        no_cookie_requests: bool = False,
    ) -> None:
        self.image = image
        self.base = f"https://{image.host}"
        self.auth = load_docker_auth(image.host, authfile)
        self.micr_realm = micr_realm
        self.no_cookie_requests = no_cookie_requests
        self.session = make_session()
        self._realm: Optional[str] = None
        self._service: Optional[str] = None
        self._tokens: Dict[str, str] = {}

        if image.host == "micr.cloud.mioffice.cn":
            self._realm = micr_realm
            self._service = "harbor-registry"

    def _discover_token_service(self) -> None:
        if self._realm:
            return
        urls = [
            f"{self.base}/v2/",
            f"{self.base}/v2/{self.image.repo}/manifests/{self.image.ref}",
        ]
        for url in urls:
            response = self.session.get(
                url,
                headers={"Accept": MANIFEST_ACCEPT},
                timeout=60,
            )
            parts = parse_www_authenticate(response.headers.get("www-authenticate"))
            if parts:
                self._realm = parts.get("realm")
                self._service = parts.get("service")
                return

    def token(self, scope: str) -> Optional[str]:
        self._discover_token_service()
        if not self._realm:
            return None
        key = f"{self._realm}|{self._service}|{scope}"
        if key in self._tokens:
            return self._tokens[key]

        params = {}
        if self._service:
            params["service"] = self._service
        if scope:
            params["scope"] = scope
        response = requests.get(
            self._realm,
            params=params,
            auth=self.auth,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("token") or payload.get("access_token")
        if not token:
            raise RuntimeError(f"token service returned no token for {self.image.host}")
        self._tokens[key] = token
        return token

    def _request(
        self,
        method: str,
        url: str,
        *,
        scope: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
        data=None,
        timeout: int = 300,
        allow_redirects: bool = False,
    ) -> requests.Response:
        headers = dict(headers or {})
        auth = self.auth
        if scope:
            token = self.token(scope)
            if token:
                headers["Authorization"] = f"Bearer {token}"
                auth = None

        # Fresh sessions avoid carrying Harbor sid cookies into upload requests.
        session = make_session() if self.no_cookie_requests else self.session
        response = session.request(
            method,
            url,
            headers=headers,
            auth=auth,
            stream=stream,
            data=data,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )

        if response.status_code == 401:
            parts = parse_www_authenticate(response.headers.get("www-authenticate"))
            if parts:
                scope = parts.get("scope") or scope or ""
                realm = parts.get("realm")
                service = parts.get("service")
                params = {}
                if service:
                    params["service"] = service
                if scope:
                    params["scope"] = scope
                token_response = requests.get(
                    realm,
                    params=params,
                    auth=self.auth,
                    timeout=60,
                )
                token_response.raise_for_status()
                token = token_response.json().get("token") or token_response.json().get(
                    "access_token"
                )
                headers["Authorization"] = f"Bearer {token}"
                response = session.request(
                    method,
                    url,
                    headers=headers,
                    auth=None,
                    stream=stream,
                    data=data,
                    timeout=timeout,
                    allow_redirects=allow_redirects,
                )
        return response

    def get_manifest(self) -> Tuple[bytes, Optional[str], Optional[str]]:
        scope = f"repository:{self.image.repo}:pull"
        response = self._request(
            "GET",
            f"{self.base}/v2/{self.image.repo}/manifests/{self.image.ref}",
            scope=scope,
            headers={"Accept": MANIFEST_ACCEPT},
            timeout=300,
        )
        response.raise_for_status()
        return (
            response.content,
            response.headers.get("Docker-Content-Digest"),
            response.headers.get("Content-Type"),
        )

    def get_blob_stream(self, digest: str) -> requests.Response:
        scope = f"repository:{self.image.repo}:pull"
        response = self._request(
            "GET",
            f"{self.base}/v2/{self.image.repo}/blobs/{digest}",
            scope=scope,
            stream=True,
            timeout=300,
            allow_redirects=True,
        )
        response.raise_for_status()
        return response

    def head_blob(self, digest: str) -> bool:
        scope = f"repository:{self.image.repo}:pull"
        response = self._request(
            "HEAD",
            f"{self.base}/v2/{self.image.repo}/blobs/{digest}",
            scope=scope,
            timeout=120,
        )
        return response.status_code in (200, 302, 307)

    def start_upload(self) -> str:
        scope = f"repository:{self.image.repo}:pull,push"
        response = self._request(
            "POST",
            f"{self.base}/v2/{self.image.repo}/blobs/uploads/",
            scope=scope,
            timeout=120,
        )
        if response.status_code != 202:
            raise RuntimeError(
                f"start upload failed {response.status_code}: {response.text[:500]}"
            )
        return response.headers["Location"]

    def put_upload_stream(
        self,
        location: str,
        digest: str,
        chunks: Iterable[bytes],
        *,
        size: int,
        media_type: str = "application/octet-stream",
    ) -> Optional[str]:
        scope = f"repository:{self.image.repo}:pull,push"
        sep = "&" if "?" in location else "?"
        response = self._request(
            "PUT",
            location + sep + urlencode({"digest": digest}),
            scope=scope,
            headers={
                "Content-Type": media_type,
                "Content-Length": str(size),
            },
            data=chunks,
            timeout=7200,
        )
        if response.status_code not in (200, 201, 202):
            raise RuntimeError(
                f"put upload failed {response.status_code}: {response.text[:500]}"
            )
        return response.headers.get("Docker-Content-Digest")

    def put_manifest(self, content: bytes, content_type: str) -> Optional[str]:
        scope = f"repository:{self.image.repo}:pull,push"
        response = self._request(
            "PUT",
            f"{self.base}/v2/{self.image.repo}/manifests/{self.image.ref}",
            scope=scope,
            headers={"Content-Type": content_type},
            data=content,
            timeout=300,
        )
        if response.status_code not in (200, 201, 202):
            raise RuntimeError(
                f"put manifest failed {response.status_code}: {response.text[:500]}"
            )
        return response.headers.get("Docker-Content-Digest")


def progress_chunks(
    response: requests.Response,
    *,
    label: str,
    size: int,
    chunk_size: int = 4 * 1024 * 1024,
    interval: int = 30,
) -> Iterable[bytes]:
    copied = 0
    last = time.time()
    for chunk in response.iter_content(chunk_size=chunk_size):
        if not chunk:
            continue
        copied += len(chunk)
        now = time.time()
        if now - last >= interval:
            print(
                f"  ... {label} streamed {copied / 1e9:.2f}/{size / 1e9:.2f} GB",
                flush=True,
            )
            last = now
        yield chunk


def copy_image(args: argparse.Namespace) -> int:
    src = ImageRef.parse(args.src)
    dst = ImageRef.parse(args.dst)

    src_client = RegistryClient(src, args.authfile, micr_realm=args.micr_realm)
    dst_client = RegistryClient(
        dst,
        args.authfile,
        micr_realm=args.micr_realm,
        no_cookie_requests=True,
    )

    print(f"[manifest] GET {src}", flush=True)
    manifest_bytes, source_digest, content_type = src_client.get_manifest()
    manifest = json.loads(manifest_bytes)
    media_type = content_type or manifest.get("mediaType")

    if "layers" not in manifest or "config" not in manifest:
        raise RuntimeError(
            "only single-arch image manifests are supported; "
            f"got mediaType={manifest.get('mediaType')}"
        )

    layers = manifest["layers"]
    blobs = [manifest["config"]] + layers
    compressed_size = sum(item.get("size", 0) for item in layers)
    print(
        "[manifest] "
        f"source_digest={source_digest} type={media_type} "
        f"layers={len(layers)} compressed_GB={compressed_size / 1e9:.3f}",
        flush=True,
    )

    uploaded = 0
    skipped = 0
    uploaded_bytes = 0
    for index, desc in enumerate(blobs):
        digest = desc["digest"]
        size = int(desc.get("size") or 0)
        label = "config" if index == 0 else f"layer {index}/{len(layers)}"

        if dst_client.head_blob(digest):
            skipped += 1
            print(f"[skip] {label} exists {digest} size={size}", flush=True)
            continue

        print(f"[copy] {label} {digest} size={size}", flush=True)
        upload_location = dst_client.start_upload()
        source_blob = src_client.get_blob_stream(digest)
        copied_digest = dst_client.put_upload_stream(
            upload_location,
            digest,
            progress_chunks(source_blob, label=label, size=size),
            size=size,
            media_type=desc.get("mediaType", "application/octet-stream"),
        )
        uploaded += 1
        uploaded_bytes += size
        print(f"[done] {label} uploaded got={copied_digest}", flush=True)

    print(f"[manifest] PUT {dst}", flush=True)
    target_digest = dst_client.put_manifest(
        manifest_bytes,
        media_type or "application/vnd.docker.distribution.manifest.v2+json",
    )
    print(f"[done] target_manifest_digest={target_digest}", flush=True)

    if not args.no_verify:
        target_bytes, verify_digest, verify_type = dst_client.get_manifest()
        target_manifest = json.loads(target_bytes)
        target_layers = target_manifest.get("layers", [])
        target_size = sum(item.get("size", 0) for item in target_layers)
        print(
            "[verify] "
            f"digest={verify_digest} type={verify_type} "
            f"layers={len(target_layers)} compressed_GB={target_size / 1e9:.3f} "
            f"manifest_bytes_equal={target_bytes == manifest_bytes}",
            flush=True,
        )
        if target_bytes != manifest_bytes:
            raise RuntimeError("target manifest bytes differ from source manifest")

    print(
        "[summary] "
        f"skipped={skipped} uploaded={uploaded} uploaded_bytes={uploaded_bytes} "
        f"source_manifest={source_digest}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a single-arch image between registries without docker pull."
    )
    parser.add_argument("--src", required=True, help="source image ref")
    parser.add_argument("--dst", required=True, help="destination image ref")
    parser.add_argument(
        "--authfile",
        default="~/.docker/config.json",
        help="Docker auth config path",
    )
    parser.add_argument(
        "--micr-realm",
        default=os.environ.get("MICR_TOKEN_REALM", DEFAULT_MICR_REALM),
        help="CloudML Harbor token realm",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip final target manifest byte comparison",
    )
    args = parser.parse_args()
    return copy_image(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
