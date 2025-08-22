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
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
              pkgs.bzip2
              pkgs.openssl
            ]}:$LD_LIBRARY_PATH
            
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              python -m venv .venv
            fi
            source .venv/bin/activate
            
            if [ ! -f .venv/installed ]; then
              echo "Installing dependencies..."
              pip install -q voyageai chromadb langchain-text-splitters numpy tqdm python-dotenv gitpython tiktoken typer pydantic
              pip install -q -e .
              touch .venv/installed
            fi
            
            export VOYAGE_API_KEY=$(cat .env | grep VOYAGE_API_KEY | cut -d'=' -f2)
            echo "🚀 Voyage embedding environment ready!"
            echo "📊 Free tokens available: 200M"
            echo "🔑 API key loaded: ''${VOYAGE_API_KEY:0:10}..."
          '';
        };
      });
}
