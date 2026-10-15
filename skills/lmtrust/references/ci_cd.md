# CI/CD Integration — Automated Quality Gate & SARIF

## GitHub Actions Workflow (`.github/workflows/lmtrust.yml`)

This workflow automatically audits every Pull Request and pushed commit, checks for logical bugs, inconsistencies, and vulnerabilities, and uploads the results directly to GitHub's **Security -> Code Scanning Alerts** tab.

```yaml
name: LMTrust Quality & Security Gate

on:
  push:
    branches: [ main, master, develop ]
  pull_request:
    branches: [ main, master ]

jobs:
  lmtrust-audit:
    name: LMTrust Logic & Security Audit
    runs-on: ubuntu-latest

    permissions:
      contents: read
      security-events: write

    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install LMTrust
        run: |
          pip install --upgrade pip
          pip install -e .

      - name: Run LMTrust Review (Diff-Aware SARIF)
        run: |
          lmtrust review --against origin/main --format sarif -o lmtrust-findings.sarif || true

      - name: Upload SARIF to GitHub Code Scanning
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: lmtrust-findings.sarif
          category: lmtrust-audit
```
