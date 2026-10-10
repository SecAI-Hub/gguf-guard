FROM docker.io/library/golang:1.27.2-alpine3.24@sha256:f92b6ef800e499660581efdabdf25d9d817a9d124eaf900924f0504e7e27e12d AS build
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
