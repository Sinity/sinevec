{
  description = "Sinevec: contextual embeddings on Qdrant";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    let
      perSystem = flake-utils.lib.eachDefaultSystem (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonPackages = pkgs.python312Packages;
          graphragPython = pkgs.python311;

          voyageaiPkg = pythonPackages.buildPythonPackage rec {
            pname = "voyageai";
            version = "0.3.4";
            src = pkgs.fetchPypi {
              inherit pname version;
              sha256 = "b2a526a31859f21782a309d8c40b69d8d8c499308ed96e2d81a6e1b952431c23";
            };
            pyproject = true;
            build-system = with pythonPackages; [
              poetry-core
            ];
            propagatedBuildInputs = with pythonPackages; [
              aiohttp
              aiolimiter
              langchain-text-splitters
              numpy
              pillow
              pydantic
              python-dotenv
              requests
              tenacity
              tokenizers
            ];
          };

          sinevecPackage = pythonPackages.buildPythonApplication {
            pname = "sinevec";
            version = "0.1.0";
            src = ./.;
            pyproject = true;
            pyprojectToml = ./pyproject.toml;
            nativeBuildInputs = with pythonPackages; [
              setuptools
              wheel
              poetry-core
            ];
            propagatedBuildInputs = with pythonPackages; [
              voyageaiPkg
              qdrant-client
              tiktoken
              python-dotenv
              typer
              pydantic
              fastapi
              uvicorn
            ];
          };

          pythonEnv = pkgs.python312.withPackages (ps:
            [
              voyageaiPkg
            ]
            ++ (with ps; [
              qdrant-client
              tiktoken
              python-dotenv
              typer
              pydantic
              fastapi
              uvicorn
              pip
              setuptools
            ])
          );

          graphragEnv = pkgs.python311.withPackages (ps:
            with ps; [
              pip
              setuptools
              wheel
            ]
          );

        in rec {
          packages.default = sinevecPackage;
          packages.sinevec = sinevecPackage;

          apps.default = {
            type = "app";
            program = "${sinevecPackage}/bin/sinevec";
          };

          apps.sinevec = apps.default;

          devShells.default = pkgs.mkShell {
            buildInputs =
              [ pythonEnv ]
              ++ (with pkgs; [
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
              ]);

            shellHook = ''
              set -euo pipefail

              export LD_LIBRARY_PATH=${
                pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.zlib
                  pkgs.bzip2
                  pkgs.openssl
                ]
              }:''${LD_LIBRARY_PATH-}

              export PYTHONPATH="$PWD/src:''${PYTHONPATH-}"
              # Add local tools to PATH (portable CLI shim)
              export PATH="$PWD/tools/bin:$PATH"

              # Load API key from .env if present, but don't override existing
              if [ -z "''${VOYAGE_API_KEY-}" ] && [ -f .env ]; then
                export VOYAGE_API_KEY="$(grep -E '^VOYAGE_API_KEY=' .env | tail -n1 | cut -d'=' -f2-)"
              fi
              if [ -z "''${OPENAI_API_KEY-}" ] && [ -f .env ]; then
                maybe_openai="$(grep -E '^OPENAI_API_KEY=' .env | tail -n1 | cut -d'=' -f2-)"
                if [ -n "$maybe_openai" ]; then
                  export OPENAI_API_KEY="$maybe_openai"
                fi
              fi

              export GRAPHRAG_ROOT="''${GRAPHRAG_ROOT-$PWD/var/graphrag}"
              if [ -z "''${GRAPHRAG_API_KEY-}" ] && [ -n "''${OPENAI_API_KEY-}" ]; then
                export GRAPHRAG_API_KEY="$OPENAI_API_KEY"
              fi
              mkdir -p "$GRAPHRAG_ROOT"
              export GRAPHRAG_VENV="$GRAPHRAG_ROOT/.venv"

              if [ ! -x "$GRAPHRAG_VENV/bin/python" ]; then
                echo "📦 Setting up GraphRAG virtualenv under $GRAPHRAG_VENV"
                ${graphragPython}/bin/python3.11 -m venv "$GRAPHRAG_VENV"
                "$GRAPHRAG_VENV/bin/pip" install --upgrade pip >/dev/null
                "$GRAPHRAG_VENV/bin/pip" install "graphrag==2.7.0" >/dev/null
              fi

              export PATH="$GRAPHRAG_VENV/bin:$PATH"

              echo "🚀 Voyage embedding environment ready!"
              if [ -n "''${VOYAGE_API_KEY-}" ]; then
                echo "🔑 API key loaded: ''${VOYAGE_API_KEY:0:10}..."
              else
                echo "🔑 Tip: add VOYAGE_API_KEY to .env for embeddings"
              fi
              if command -v graphrag >/dev/null 2>&1; then
                echo "🕸️  GraphRAG CLI available (root: $GRAPHRAG_ROOT)"
              fi
            '';
          };

          devShells.graphrag = pkgs.mkShell {
            buildInputs =
              [ graphragEnv ]
              ++ (with pkgs; [
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
              ]);

            shellHook = ''
              set -euo pipefail

              export LD_LIBRARY_PATH=${
                pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.zlib
                  pkgs.bzip2
                  pkgs.openssl
                ]
              }:''${LD_LIBRARY_PATH-}

              export PYTHONPATH="$PWD/src:''${PYTHONPATH-}"
              export PATH="$PWD/tools/bin:$PATH"

              export GRAPHRAG_ROOT="''${GRAPHRAG_ROOT-$PWD/var/graphrag}"
              mkdir -p "''${GRAPHRAG_ROOT}"
              export GRAPHRAG_VENV="$GRAPHRAG_ROOT/.venv"

              if [ -z "''${OPENAI_API_KEY-}" ] && [ -f .env ]; then
                maybe_key="$(grep -E '^OPENAI_API_KEY=' .env | tail -n1 | cut -d'=' -f2-)"
                if [ -n "$maybe_key" ]; then
                  export OPENAI_API_KEY="$maybe_key"
                fi
              fi

              if [ -z "''${GRAPHRAG_API_KEY-}" ] && [ -n "''${OPENAI_API_KEY-}" ]; then
                export GRAPHRAG_API_KEY="$OPENAI_API_KEY"
              fi

              if [ ! -x "$GRAPHRAG_VENV/bin/python" ]; then
                echo "📦 Installing GraphRAG CLI under $GRAPHRAG_VENV"
                ${graphragPython}/bin/python3.11 -m venv "$GRAPHRAG_VENV"
                "$GRAPHRAG_VENV/bin/pip" install --upgrade pip >/dev/null
                "$GRAPHRAG_VENV/bin/pip" install "graphrag==2.7.0" >/dev/null
              fi

              export PATH="$GRAPHRAG_VENV/bin:$PATH"

              echo "🕸️  GraphRAG environment ready (root: $GRAPHRAG_ROOT)"
              if [ -n "''${OPENAI_API_KEY-}" ]; then
                echo "🔑 OPENAI_API_KEY available"
              fi
              if [ -n "''${GRAPHRAG_API_KEY-}" ]; then
                echo "🔑 GRAPHRAG_API_KEY set (defaults to OPENAI_API_KEY)"
              else
                echo "🔑 Set OPENAI_API_KEY or Azure equivalents before indexing"
              fi
              if command -v graphrag >/dev/null 2>&1; then
                echo "🧠 GraphRAG CLI available via $(command -v graphrag)"
              fi
            '';
          };
        });

      sinevecModule = { config, lib, pkgs, ... }@args:
        import ./nixos/sinevec-service.nix (args // { inherit self; });
    in
    perSystem
    // {
      nixosModules = rec {
        default = sinevecModule;
        sinevec = default;
      };
    };
}
