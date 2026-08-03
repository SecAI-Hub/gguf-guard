package analysis

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func decodeJSONFile(path string, maxBytes int64, dst any) error {
	pathInfo, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !pathInfo.Mode().IsRegular() {
		return fmt.Errorf("refusing non-regular JSON input: %s", pathInfo.Mode())
	}
	if pathInfo.Size() > maxBytes {
		return fmt.Errorf("JSON input exceeds %d-byte limit", maxBytes)
	}

	// #nosec G304 -- this is the centralized bounded reader for an explicit
	// caller-selected sidecar; Lstat/Fstat/path-after identity checks surround it.
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	openedInfo, err := f.Stat()
	if err != nil {
		return err
	}
	if !os.SameFile(pathInfo, openedInfo) || openedInfo.Size() != pathInfo.Size() {
		return fmt.Errorf("JSON input changed while opening")
	}

	data, err := io.ReadAll(io.LimitReader(f, maxBytes+1))
	if err != nil {
		return err
	}
	if int64(len(data)) > maxBytes {
		return fmt.Errorf("JSON input exceeds %d-byte limit", maxBytes)
	}
	after, err := f.Stat()
	if err != nil || !os.SameFile(pathInfo, after) || after.Size() != pathInfo.Size() || !after.ModTime().Equal(pathInfo.ModTime()) {
		return fmt.Errorf("JSON input changed while reading")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return fmt.Errorf("unexpected trailing JSON value")
		}
		return fmt.Errorf("trailing JSON data: %w", err)
	}
	return nil
}

func writeJSONAtomic(path string, value any, mode os.FileMode) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')

	if info, err := os.Lstat(path); err == nil {
		if !info.Mode().IsRegular() {
			return fmt.Errorf("refusing to replace non-regular output: %s", info.Mode())
		}
	} else if !os.IsNotExist(err) {
		return err
	}

	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	if err := tmp.Chmod(mode); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	// #nosec G304 -- dir is the parent of the caller-selected output and is
	// opened only to fsync the already completed atomic rename.
	dirHandle, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer dirHandle.Close()
	if err := dirHandle.Sync(); err != nil {
		return err
	}
	return nil
}
