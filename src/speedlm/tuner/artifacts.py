"""Content-addressed immutable EAGLE-3 draft artifact registry."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speedlm.storage import atomic_write_json

_MANIFEST_NAME = "manifest.json"


class ArtifactError(RuntimeError):
    """Raised when an artifact is invalid or cannot be published safely."""


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Provenance supplied when materializing a trained draft."""

    verifier_model: str
    draft_model: str
    base_draft: str
    trace_hash: str
    training_params: Mapping[str, object]
    val_loss: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("verifier_model", self.verifier_model),
            ("draft_model", self.draft_model),
            ("base_draft", self.base_draft),
            ("trace_hash", self.trace_hash),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Stored manifest for one immutable draft directory."""

    artifact_id: str
    verifier_model: str
    draft_model: str
    base_draft: str
    trace_hash: str
    training_params: Mapping[str, object]
    created_at: float
    val_loss: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "verifier_model": self.verifier_model,
            "draft_model": self.draft_model,
            "base_draft": self.base_draft,
            "trace_hash": self.trace_hash,
            "training_params": dict(self.training_params),
            "created_at": self.created_at,
            "val_loss": self.val_loss,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactManifest:
        required = {
            "artifact_id",
            "verifier_model",
            "draft_model",
            "base_draft",
            "trace_hash",
            "training_params",
            "created_at",
        }
        # val_loss is optional (added after initial release); allow it as
        # an extra key but don't require it.
        allowed = required | {"val_loss"}
        if not required.issubset(set(value)):
            raise ArtifactError("artifact manifest has missing fields")
        extra = set(value) - allowed
        if extra:
            raise ArtifactError(f"artifact manifest has unknown fields: {extra}")
        params = value["training_params"]
        created_at = value["created_at"]
        if not isinstance(params, dict):
            raise ArtifactError("manifest training_params must be an object")
        if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
            raise ArtifactError("manifest created_at must be numeric")
        strings: dict[str, str] = {}
        for fld in (
            "artifact_id",
            "verifier_model",
            "draft_model",
            "base_draft",
            "trace_hash",
        ):
            item = value[fld]
            if not isinstance(item, str) or not item:
                raise ArtifactError(f"manifest {fld} must be a non-empty string")
            strings[fld] = item
        val_loss = value.get("val_loss")
        if val_loss is not None:
            if isinstance(val_loss, bool) or not isinstance(val_loss, (int, float)):
                raise ArtifactError("manifest val_loss must be a number or null")
            val_loss = float(val_loss)
        return cls(
            artifact_id=strings["artifact_id"],
            verifier_model=strings["verifier_model"],
            draft_model=strings["draft_model"],
            base_draft=strings["base_draft"],
            trace_hash=strings["trace_hash"],
            training_params=params,
            created_at=float(created_at),
            val_loss=val_loss,
        )


@dataclass(frozen=True, slots=True)
class Artifact:
    """A published artifact and its verified manifest."""

    path: Path
    manifest: ArtifactManifest

    @property
    def artifact_id(self) -> str:
        return self.manifest.artifact_id


@dataclass(frozen=True, slots=True)
class ActivePointer:
    """Current artifact plus prior artifacts available for rollback."""

    artifact_id: str
    history: tuple[str, ...]
    updated_at: float

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "history": list(self.history),
            "updated_at": self.updated_at,
        }


def hash_directory(path: Path) -> str:
    """Hash regular file paths and bytes in deterministic lexical order."""
    if not path.is_dir():
        raise ArtifactError(f"artifact source is not a directory: {path}")
    digest = hashlib.sha256()
    files = sorted(
        (
            entry
            for entry in path.rglob("*")
            if entry.relative_to(path).as_posix() != _MANIFEST_NAME
        ),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    for entry in files:
        relative = entry.relative_to(path).as_posix()
        if entry.is_symlink():
            raise ArtifactError(f"artifact may not contain symlinks: {relative}")
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise ArtifactError(f"artifact contains a non-regular file: {relative}")
        digest.update(b"file\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        try:
            with entry.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise ArtifactError(f"cannot hash artifact file: {entry}") from exc
        digest.update(b"\0")
    return digest.hexdigest()


class ArtifactRegistry:
    """Publish immutable artifacts and atomically manage the active pointer."""

    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], float] = time.time,
        before_publish: Callable[[Path], None] | None = None,
    ) -> None:
        self._root = root
        self._artifacts_dir = root / "artifacts"
        self._active_path = root / "active.json"
        self._clock = clock
        self._before_publish = before_publish
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active_path(self) -> Path:
        return self._active_path

    def publish(self, source: Path, spec: ArtifactSpec) -> Artifact:
        """Copy *source* into a same-filesystem temp dir, then rename atomically."""
        if (source / _MANIFEST_NAME).exists():
            raise ArtifactError(f"source may not contain reserved {_MANIFEST_NAME}")
        artifact_id = hash_directory(source)
        target = self._artifacts_dir / artifact_id
        if target.exists():
            return self.get(artifact_id)

        temp_path = Path(
            tempfile.mkdtemp(prefix=f".publish-{artifact_id[:12]}-", dir=self._artifacts_dir)
        )
        committed = False
        try:
            shutil.copytree(source, temp_path, dirs_exist_ok=True, symlinks=False)
            # copytree's trailing copystat stamps the source's mode onto
            # temp_path, so a sealed 0555 draft would leave the temp tree
            # read-only and the manifest sidecar below could never be created.
            # Restoring owner-write cannot change the artifact ID (mode is not
            # an input to hash_directory) and cannot change the published
            # permissions (_make_tree_read_only re-seals before the rename).
            _make_tree_writable(temp_path)
            copied_hash = hash_directory(temp_path)
            if copied_hash != artifact_id:
                raise ArtifactError(
                    f"artifact changed while publishing: expected {artifact_id}, got {copied_hash}"
                )
            manifest = ArtifactManifest(
                artifact_id=artifact_id,
                verifier_model=spec.verifier_model,
                draft_model=spec.draft_model,
                base_draft=spec.base_draft,
                trace_hash=spec.trace_hash,
                training_params=dict(spec.training_params),
                created_at=self._clock(),
                val_loss=spec.val_loss,
            )
            atomic_write_json(temp_path / _MANIFEST_NAME, manifest.to_dict())
            _make_tree_read_only(temp_path)
            if self._before_publish is not None:
                self._before_publish(temp_path)
            try:
                os.rename(temp_path, target)
                committed = True
            except FileExistsError:
                return self.get(artifact_id)
        except (OSError, ValueError, TypeError) as exc:
            # ArtifactError subclasses RuntimeError and so is never caught here;
            # "artifact changed while publishing" propagates uncaught.
            raise ArtifactError(f"cannot publish artifact {artifact_id}: {exc!r}") from exc
        finally:
            if not committed and temp_path.exists():
                _make_tree_writable(temp_path)
                shutil.rmtree(temp_path, ignore_errors=True)
        # Deliberately not ``self.get(artifact_id)``: that would be a third full
        # SHA-256 pass over the tree, on top of the source hash and the
        # post-copy verification hash above.  For a 2 GB EAGLE-3 draft each pass
        # is ~2 GB of reads, and all three run while the gateway's admission
        # gate is closed -- so the redundancy was paid for in serving downtime.
        #
        # The integrity guarantee is unchanged, because the dropped pass never
        # added one.  ``copied_hash`` already proved that this exact tree hashes
        # to ``artifact_id``; ``_make_tree_read_only`` then sealed it, and
        # ``os.rename`` is atomic and moves the very inode that was verified.
        # Nothing in between can alter content: the manifest sidecar is excluded
        # from the hash by :func:`hash_directory`, and permissions are not
        # hashed.  ``get`` still re-hashes on every *later* read, which is where
        # a bit-rot or tamper check belongs.
        return Artifact(path=target, manifest=manifest)

    def get(self, artifact_id: str) -> Artifact:
        """Load and content-verify a published artifact."""
        _validate_artifact_id(artifact_id)
        path = self._artifacts_dir / artifact_id
        manifest_path = path / _MANIFEST_NAME
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read artifact manifest: {manifest_path}") from exc
        if not isinstance(raw, dict):
            raise ArtifactError("artifact manifest must contain an object")
        manifest = ArtifactManifest.from_dict(raw)
        if manifest.artifact_id != artifact_id:
            raise ArtifactError("artifact directory and manifest IDs differ")
        actual = hash_directory(path)
        if actual != artifact_id:
            raise ArtifactError(
                f"artifact content hash mismatch: expected {artifact_id}, got {actual}"
            )
        return Artifact(path=path, manifest=manifest)

    def active(self) -> Artifact | None:
        pointer = self.active_pointer()
        if pointer is None:
            return None
        return self.get(pointer.artifact_id)

    def active_pointer(self) -> ActivePointer | None:
        if not self._active_path.exists():
            return None
        try:
            raw = json.loads(self._active_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read active pointer: {self._active_path}") from exc
        if not isinstance(raw, dict):
            raise ArtifactError("active pointer must contain an object")
        if set(raw) != {"artifact_id", "history", "updated_at"}:
            raise ArtifactError("active pointer has missing or unknown fields")
        artifact_id = raw["artifact_id"]
        history = raw["history"]
        updated_at = raw["updated_at"]
        if not isinstance(artifact_id, str):
            raise ArtifactError("active artifact_id must be a string")
        if not isinstance(history, list) or not all(isinstance(item, str) for item in history):
            raise ArtifactError("active history must be a list of strings")
        if isinstance(updated_at, bool) or not isinstance(updated_at, (int, float)):
            raise ArtifactError("active updated_at must be numeric")
        _validate_artifact_id(artifact_id)
        for item in history:
            _validate_artifact_id(item)
        return ActivePointer(
            artifact_id=artifact_id,
            history=tuple(history),
            updated_at=float(updated_at),
        )

    def promote(self, artifact_id: str, *, gate_passed: bool) -> ActivePointer:
        """Atomically activate an artifact, but only with an explicit gate pass."""
        if gate_passed is not True:
            raise ArtifactError("refusing to activate artifact without a gate pass")
        self.get(artifact_id)
        current = self.active_pointer()
        history = current.history if current is not None else ()
        if current is not None and current.artifact_id != artifact_id:
            history = (*history, current.artifact_id)
        pointer = ActivePointer(
            artifact_id=artifact_id,
            history=history,
            updated_at=self._clock(),
        )
        atomic_write_json(self._active_path, pointer.to_dict())
        return pointer

    def rollback(self) -> ActivePointer | None:
        """Atomically restore the previous active artifact, if one exists."""
        current = self.active_pointer()
        if current is None or not current.history:
            return current
        previous = current.history[-1]
        self.get(previous)
        pointer = ActivePointer(
            artifact_id=previous,
            history=current.history[:-1],
            updated_at=self._clock(),
        )
        atomic_write_json(self._active_path, pointer.to_dict())
        return pointer


def _validate_artifact_id(artifact_id: str) -> None:
    if len(artifact_id) != 64:
        raise ArtifactError("artifact ID must be a SHA-256 hex digest")
    try:
        bytes.fromhex(artifact_id)
    except ValueError as exc:
        raise ArtifactError("artifact ID must be a SHA-256 hex digest") from exc


def _make_tree_read_only(path: Path) -> None:
    for entry in sorted(path.rglob("*"), reverse=True):
        if entry.is_dir():
            entry.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
        else:
            entry.chmod(stat.S_IRUSR | stat.S_IRGRP)
    path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)


def _make_tree_writable(path: Path) -> None:
    for entry in path.rglob("*"):
        if entry.is_dir():
            entry.chmod(stat.S_IRWXU)
        else:
            entry.chmod(stat.S_IRUSR | stat.S_IWUSR)
    path.chmod(stat.S_IRWXU)
