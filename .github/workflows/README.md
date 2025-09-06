# Publishing with Git Tags

## PyPI (Production)
```bash
git tag v1.0.0
git push origin v1.0.0
```

## TestPyPI (Testing)
```bash
git tag v1.0.0-test
git push origin v1.0.0-test
```

# Tag Rules

- **Production**: `v1.0.0`, `v2.1.3` (no `-test` suffix)
- **Testing**: `v1.0.0-test`, `v2.1.3-test` (with `-test` suffix)

# Common Commands

```bash
# List tags
git tag -l

# Delete tag locally and remotely
git tag -d v1.0.0
git push origin --delete v1.0.0
```

The GitHub Actions automatically detect the tag format and publish to the appropriate repository.