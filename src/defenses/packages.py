"""Complete researcher-supplied packages; no package content is generated here."""
import hashlib
from pathlib import Path
import shutil


def package_files(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Package root must be a directory, not a symlink')
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Package symlinks are not supported')
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError('Package contains a non-regular filesystem object')
    if 'SKILL.md' not in files:
        raise ValueError('Package requires a top-level SKILL.md')
    if any(p.endswith('/SKILL.md') for p in files):
        raise ValueError('Nested skill packages are ambiguous')
    return files


def package_hash(root):
    root = Path(root)
    digest = hashlib.sha256()
    for relative in package_files(root):
        name, content = relative.encode(), (root / relative).read_bytes()
        digest.update(len(name).to_bytes(8, 'big')); digest.update(name)
        digest.update(len(content).to_bytes(8, 'big')); digest.update(content)
    return digest.hexdigest()


def copy_full_package(root, destination, expected_hash):
    root, destination = Path(root), Path(destination)
    files = package_files(root)
    if package_hash(root) != expected_hash:
        raise ValueError('Package changed before scanning')
    # Preserve any racing symlink so validation rejects it instead of following it.
    shutil.copytree(root, destination, symlinks=True)
    if package_hash(destination) != expected_hash:
        raise ValueError('Copied package hash mismatch')
    return destination, files
