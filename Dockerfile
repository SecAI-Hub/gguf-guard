FROM docker.io/library/golang:1.27.1-alpine3.24@sha256:8a5910f31396cd4d89662f56c68b3ae31d374308270a1c3bd96672ee5ed43414 AS build
WORKDIR /src
COPY go.mod ./
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gguf-guard ./cmd/gguf-guard

FROM docker.io/library/alpine:3.24.1@sha256:28bd5fe8b56d1bd048e5babf5b10710ebe0bae67db86916198a6eec434943f8b
RUN mkdir -p /work && chown 65534:65534 /work
COPY --from=build /out/gguf-guard /usr/local/bin/gguf-guard
USER 65534:65534
WORKDIR /work
ENTRYPOINT ["gguf-guard"]
