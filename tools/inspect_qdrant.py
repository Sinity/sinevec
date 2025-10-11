#!/usr/bin/env python
from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from sinevec.embed_utils import (
    QDRANT_API_KEY,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
    QDRANT_HTTPS,
)


def main() -> None:
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_HTTP_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        api_key=QDRANT_API_KEY,
        https=QDRANT_HTTPS,
    )
    info = client.get_collections()
    if not info.collections:
        print("No collections found.")
        return
    for col in info.collections:
        print(f"\nCollection: {col.name}")
        try:
            count = client.count(col.name, exact=True).count
        except Exception as exc:  # pragma: no cover - diagnostic helper
            print(f"  count: error ({exc})")
        else:
            print(f"  count: {count}")

        params: Any = getattr(getattr(col, "config", None), "params", None)
        vectors = getattr(params, "vectors", None)
        if vectors and getattr(vectors, "size", None):
            print(f"  vector_size: {vectors.size}")


if __name__ == "__main__":  # pragma: no cover
    main()
