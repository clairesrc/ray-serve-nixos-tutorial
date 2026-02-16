{
  description = "Ray Serve Tutorial Environment";

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
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            (python3.withPackages (
              ps: with ps; [
                ray
                requests
                starlette
                starlette
                uvicorn
                grpcio
                pydantic
                fastapi
                aiosignal
                frozenlist
                jsonschema
                pyyaml
                watchfiles
                torch
                aiohttp
                hydra-core
                omegaconf
                prometheus-client
                aiohttp-cors
                opencensus
                gpustat
                aiorwlock
                colorful
                setproctitle
                smart-open
                psutil
                msgpack
                opentelemetry-api
                opentelemetry-sdk
                opentelemetry-exporter-prometheus
                opentelemetry-proto
              ]
            ))
          ];
        };
      }
    );
}
