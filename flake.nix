{
  description = "Voyage Embeddings for Everything";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonPackages = pkgs.python311Packages;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python311
            pythonPackages.pip
            pythonPackages.virtualenv
            git
            jq
            ripgrep
            fd
            unzip
            gcc
            stdenv.cc.cc.lib
            zlib
            bzip2
            openssl
          ];

          shellHook = ''
            set -euo pipefail

            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
              pkgs.bzip2
              pkgs.openssl
            ]}:''${LD_LIBRARY_PATH-}
            
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              python -m venv .venv
            fi
            source .venv/bin/activate
            
            # Ensure dependencies and project are installed once
            if [ ! -f .venv/installed ]; then
              echo "Installing dependencies and project (editable)..."
              pip install -q --upgrade pip
              pip install -q voyageai chromadb langchain-text-splitters numpy tqdm python-dotenv gitpython tiktoken typer pydantic
              pip install -q -e .
              touch .venv/installed || true
            fi
            
            # Add local tools to PATH (portable CLI shim)
            export PATH="$PWD/tools/bin:$PATH"

            # Load API key from .env if present, but don't override existing
            if [ -z "''${VOYAGE_API_KEY-}" ] && [ -f .env ]; then
              export VOYAGE_API_KEY="$(grep -E '^VOYAGE_API_KEY=' .env | tail -n1 | cut -d'=' -f2-)"
            fi

            echo "🚀 Voyage embedding environment ready!"
            echo "📊 Free tokens available: 200M"
            if [ -n "''${VOYAGE_API_KEY-}" ]; then
              echo "🔑 API key loaded: ''${VOYAGE_API_KEY:0:10}..."
            else
              echo "🔑 Tip: add VOYAGE_API_KEY to .env for embeddings"
            fi
          '';
        };
        # Provide a nix app for one-liner execution
        apps.ve = {
          type = "app";
          program = (pkgs.writeShellApplication {
            name = "ve";
            runtimeInputs = [ pkgs.python311 pkgs.python311Packages.pip pkgs.python311Packages.virtualenv ];
            text = ''
              set -euo pipefail
              if [ -z "''${VIRTUAL_ENV-}" ] && [ -f .venv/bin/activate ]; then
                # shellcheck disable=SC1091
                source .venv/bin/activate
              fi
              export PYTHONPATH="$PWD/src:''${PYTHONPATH-}"
              if [ -z "''${VOYAGE_API_KEY-}" ] && [ -f .env ]; then
                export VOYAGE_API_KEY="$(grep -E '^VOYAGE_API_KEY=' .env | tail -n1 | cut -d'=' -f2-)"
              fi
              python - "$@" <<'PY'
from voyage_embeddings.cli import app
app(prog_name="ve")
PY
            '';
          }).outPath + "/bin/ve";
        };
      });
}
