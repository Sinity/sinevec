{ lib, pkgs, config, self, ... }:
let
  inherit (lib)
    mkEnableOption
    mkIf
    mkOption
    mkDefault
    mkMerge
    unique
    optional
    optionalAttrs
    types
    escapeShellArgs
    hasPrefix;

  cfg = config.services.sinevec;

  defaultPackage =
    self.packages.${pkgs.stdenv.hostPlatform.system}.sinevec;

  defaultDataDir = "/var/lib/sinevec";
  defaultLogDir = "/var/log/sinevec";
  defaultStateDir = "${defaultDataDir}/state";
in
{
  options.services.sinevec = {
    enable =
      mkEnableOption "the Sinevec vector search service";

    package = mkOption {
      type = types.package;
      default = defaultPackage;
      defaultText = "self.packages.\${pkgs.stdenv.hostPlatform.system}.sinevec";
      description = ''
        Package that provides the `sinevec` CLI.
        Override when you want to run a custom build or wrapper.
      '';
    };

    host = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = ''
        Address that `sinevec serve` binds to.
        Applied automatically when the command subcommand is `serve`.
      '';
      example = "0.0.0.0";
    };

    port = mkOption {
      type = types.port;
      default = 8000;
      description = ''
        Port that `sinevec serve` listens on.
        Applied automatically when the command subcommand is `serve`.
      '';
    };

    dataDir = mkOption {
      type = types.path;
      default = defaultDataDir;
      description = ''
        Working directory for the service.
        The directory is created automatically when it does not already exist.
      '';
    };

    logDir = mkOption {
      type = types.path;
      default = defaultLogDir;
      description = ''
        Directory that receives service logs.
        Logs are appended to `sinevec.log` inside this directory.
      '';
    };

    stateDir = mkOption {
      type = types.path;
      default = defaultStateDir;
      description = ''
        Directory that stores runtime state (embedding checkpoints, progress files).
        Pipelines read this path via ``SINEVEC_STATE_DIR``.
      '';
    };

    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [ ];
      description = ''
        Additional arguments appended to the `sinevec serve` invocation.
      '';
      example = [ "--model" "voyage-3" "--json" ];
    };

    logToJournal = mkOption {
      type = types.bool;
      default = true;
      description = ''
        When true, write logs to the systemd journal. Disable to append logs
        to ``${cfg.logDir}/sinevec.log`` instead.
      '';
    };

    environment = mkOption {
      type = types.attrsOf types.str;
      default = { };
      description = "Extra environment variables for the service.";
    };

    environmentFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = ''
        Optional file containing KEY=VALUE pairs that systemd loads before
        starting the service.
      '';
    };

    environmentFiles = mkOption {
      type = types.listOf types.path;
      default = [ ];
      description = "Additional EnvironmentFile entries loaded before the service starts.";
    };

    manageUser = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Whether the module should declare the service user and group automatically.
        Disable when you want to provide the identities externally.
      '';
    };

    user = mkOption {
      type = types.str;
      default = "sinevec";
      description = "User account that runs the service.";
    };

    group = mkOption {
      type = types.str;
      default = "sinevec";
      description = "Primary group for the service user.";
    };

    qdrant = {
      host = mkOption {
        type = types.str;
        default = "127.0.0.1";
        description = "Qdrant host passed via QDRANT_HOST.";
      };

      httpPort = mkOption {
        type = types.port;
        default = 6333;
        description = "Qdrant HTTP port.";
      };

      grpcPort = mkOption {
        type = types.nullOr types.port;
        default = null;
        description = "Optional Qdrant gRPC port.";
      };

      apiKey = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Inline Qdrant API key (consider using qdrant.apiKeyFile instead).";
      };

      apiKeyFile = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Environment file exporting QDRANT_API_KEY=... (preferred over qdrant.apiKey).";
      };

      useHttps = mkOption {
        type = types.bool;
        default = false;
        description = "Set QDRANT_USE_HTTPS=1 when true.";
      };

      timeout = mkOption {
        type = types.float;
        default = 20.0;
        description = "Client timeout in seconds (QDRANT_CLIENT_TIMEOUT).";
      };

      vectorSize = mkOption {
        type = types.nullOr types.int;
        default = null;
        description = "Override QDRANT_VECTOR_SIZE when non-null.";
      };
    };

    voyage = {
      apiKey = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Inline Voyage API key (VOYAGE_API_KEY).";
      };

      apiKeyFile = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Environment file exporting VOYAGE_API_KEY=... (preferred).";
      };

      contextModel = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Override VOYAGE_CONTEXT_MODEL.";
      };

      embedModel = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Override VOYAGE_EMBED_MODEL.";
      };

      queryModel = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Override VOYAGE_QUERY_MODEL.";
      };
    };
  };

  config = mkIf cfg.enable (
    let
      environmentFilesList =
        (lib.optional (cfg.environmentFile != null) cfg.environmentFile)
        ++ cfg.environmentFiles
        ++ lib.optional (cfg.qdrant.apiKeyFile != null) cfg.qdrant.apiKeyFile
        ++ lib.optional (cfg.voyage.apiKeyFile != null) cfg.voyage.apiKeyFile;

      qdrantEnv =
        lib.filterAttrs (_: v: v != null) {
          QDRANT_HOST = cfg.qdrant.host;
          QDRANT_HTTP_PORT = builtins.toString cfg.qdrant.httpPort;
          QDRANT_GRPC_PORT = if cfg.qdrant.grpcPort != null then builtins.toString cfg.qdrant.grpcPort else null;
          QDRANT_USE_HTTPS = if cfg.qdrant.useHttps then "1" else "0";
          QDRANT_CLIENT_TIMEOUT = builtins.toString cfg.qdrant.timeout;
          QDRANT_VECTOR_SIZE = if cfg.qdrant.vectorSize != null then builtins.toString cfg.qdrant.vectorSize else null;
          QDRANT_API_KEY = cfg.qdrant.apiKey;
        };

      voyageEnv =
        lib.filterAttrs (_: v: v != null) {
          VOYAGE_API_KEY = cfg.voyage.apiKey;
          VOYAGE_CONTEXT_MODEL = cfg.voyage.contextModel;
          VOYAGE_EMBED_MODEL = cfg.voyage.embedModel;
          VOYAGE_QUERY_MODEL = cfg.voyage.queryModel;
        };

      cacheDir = "${cfg.stateDir}/cache";
    in {
      assertions = [
        {
          assertion = hasPrefix "/" (toString cfg.dataDir);
          message = "services.sinevec.dataDir must be an absolute path.";
        }
        {
          assertion = hasPrefix "/" (toString cfg.logDir);
          message = "services.sinevec.logDir must be an absolute path.";
        }
      ];

      users.groups = mkIf cfg.manageUser {
        "${cfg.group}" = { };
      };

      users.users = mkIf cfg.manageUser {
        "${cfg.user}" = {
          description = "Sinevec vector search service user";
          group = cfg.group;
          isSystemUser = true;
          home = cfg.dataDir;
          createHome = false;
        };
      };

      systemd.tmpfiles.rules =
        unique (
          [
            "d ${cfg.dataDir} 0750 ${cfg.user} ${cfg.group} -"
            "d ${cfg.stateDir} 0750 ${cfg.user} ${cfg.group} -"
            "d ${cacheDir} 0750 ${cfg.user} ${cfg.group} -"
          ]
          ++ (lib.optional (!cfg.logToJournal) "d ${cfg.logDir} 0750 ${cfg.user} ${cfg.group} -")
        );

      systemd.services.sinevec = {
        description = "Sinevec vector search service";
        wantedBy = [ "multi-user.target" ];
        after = [ "network-online.target" ];
        requires = [ "network-online.target" ];

        environment = lib.mkMerge [
          cfg.environment
          {
            SINEVEC_DATA_DIR = lib.mkDefault (toString cfg.dataDir);
            SINEVEC_DATA_ROOT = lib.mkDefault (toString cfg.dataDir);
            SINEVEC_STATE_DIR = lib.mkDefault (toString cfg.stateDir);
            SINEVEC_LOG_DIR = lib.mkDefault (toString cfg.logDir);
            XDG_DATA_HOME = lib.mkDefault (toString cfg.dataDir);
            XDG_STATE_HOME = lib.mkDefault (toString cfg.stateDir);
            XDG_CACHE_HOME = lib.mkDefault cacheDir;
          }
          qdrantEnv
          voyageEnv
        ];

        serviceConfig =
          {
            Type = "simple";
            User = cfg.user;
            Group = cfg.group;
            WorkingDirectory = toString cfg.dataDir;
            ExecStart = escapeShellArgs (
              [ (lib.getExe cfg.package)
                "serve"
                "--host"
                cfg.host
                "--port"
                (toString cfg.port)
              ]
              ++ cfg.extraArgs
            );
            Restart = "on-failure";
            RestartSec = "5s";
          }
          // (if cfg.logToJournal then {
            StandardOutput = "journal";
            StandardError = "journal";
          } else {
            StandardOutput = "append:${cfg.logDir}/sinevec.log";
            StandardError = "append:${cfg.logDir}/sinevec.log";
          })
          // optionalAttrs (environmentFilesList != []) {
            EnvironmentFile = environmentFilesList;
          };
      };
    }
  );
}
