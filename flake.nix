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
        in rec {
          packages.default = sinevecPackage;
          packages.sinevec = sinevecPackage;

          apps.default = {
            type = "app";
            program = "${sinevecPackage}/bin/sinevec";
          };

          apps.sinevec = apps.default;

          devShells.default = pkgs.mkShell {
            buildInputs = with pkgs; [
              python312
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

              export LD_LIBRARY_PATH=${
                pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.zlib
                  pkgs.bzip2
                  pkgs.openssl
                ]
              }:''${LD_LIBRARY_PATH-}

              if [ ! -d .venv ]; then
                echo "Creating virtual environment..."
                python -m venv .venv
              fi
              source .venv/bin/activate

              # Ensure dependencies and project are installed once
              if [ ! -f .venv/installed ]; then
                echo "Installing dependencies and project (editable)..."
                pip install -q --upgrade pip
                pip install -q voyageai qdrant-client python-dotenv tiktoken typer pydantic fastapi uvicorn
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
              if [ -n "''${VOYAGE_API_KEY-}" ]; then
                echo "🔑 API key loaded: ''${VOYAGE_API_KEY:0:10}..."
              else
                echo "🔑 Tip: add VOYAGE_API_KEY to .env for embeddings"
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
