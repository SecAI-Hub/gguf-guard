FROM docker.io/library/golang:1.26.6-alpine@sha256:3889b425f035be855a72fb4755265311293b6d414521f0a519d819df32222d83 AS build
WORKDIR /src
COPY go.mod ./
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gguf-guard ./cmd/gguf-guard

FROM docker.io/library/alpine:3.23@sha256:fd791d74b68913cbb027c6546007b3f0d3bc45125f797758156952bc2d6daf40
RUN mkdir -p /work && chown 65534:65534 /work
COPY --from=build /out/gguf-guard /usr/local/bin/gguf-guard
USER 65534:65534
WORKDIR /work
ENTRYPOINT ["gguf-guard"]
