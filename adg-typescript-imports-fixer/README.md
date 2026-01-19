# 🔧 ADG TypeScript Imports Fixer

An intelligent tool for analyzing and automatically fixing broken import paths in TypeScript/JavaScript projects. Perfect for VS Code extension development and large-scale refactoring.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MPL--2.0-blue.svg)

## ✨ Features

- 🔍 **Smart Scanning** - Recursively scans TypeScript/JavaScript projects for all import statements
- 📋 **tsconfig.json Aware** - Reads and respects `baseUrl`, `paths` aliases, `include`/`exclude` patterns
- 🎯 **Intelligent Matching** - Uses multiple strategies to find correct module paths
- 📊 **Confidence Levels** - Categorizes fixes by confidence (HIGH/MEDIUM/LOW) for safe incremental fixes
- 🛡️ **Dry-Run Mode** - Preview all changes before applying them
- 📁 **Index File Support** - Correctly handles `index.ts` barrel exports
- 📤 **JSON Export** - Export scan results for CI/CD integration
- ⚠️ **Issue Detection** - Detects leftover `.ts` extensions, missing files, wrong path depth

## 🚀 Quick Start

### Installation

No installation required! Just Python 3.8+:

```bash
# Clone the repository
git clone https://github.com/yourusername/adg-typescript-imports-fixer.git
cd adg-typescript-imports-fixer
```

### Basic Usage (v2 - Recommended)

```bash
# Auto-detect tsconfig.json and scan
python adg-typescript-imports-fixer2.py

# Scan specific directory
python adg-typescript-imports-fixer2.py --dir ./src

# Use specific tsconfig.json
python adg-typescript-imports-fixer2.py --tsconfig ./tsconfig.json

# Preview HIGH confidence fixes (safe)
python adg-typescript-imports-fixer2.py --fix-high --dry-run

# Apply HIGH confidence fixes
python adg-typescript-imports-fixer2.py --fix-high

# Apply all suggested fixes (review carefully!)
python adg-typescript-imports-fixer2.py --fix-all
```

## 📖 How It Works

1. **Finds tsconfig.json** - Searches from current directory upward (or uses specified path)
2. **Discovers all modules** - Builds a registry of all `.ts`, `.tsx`, `.js`, `.jsx` files
3. **Scans every file** - Finds all import/export statements
4. **Validates each import** - Tests the path FROM THE IMPORTING FILE'S LOCATION
5. **Suggests fixes** - With confidence levels based on match quality

## 🎯 Detected Issues

| Icon | Issue | Description |
|------|-------|-------------|
| 🔍 | `cannot_resolve` | Import path doesn't resolve to any file |
| 📄 | `missing_file` | Directory exists but target file was deleted/moved |
| 📎 | `has_ts_extension` | Import has `.ts`/`.tsx` extension (TypeScript anti-pattern) |
| 📏 | `wrong_depth` | Path escapes project root (too many `../`) |
| 🏷️ | `alias_not_found` | `@alias` not defined in tsconfig.json paths |
| 📁 | `index_explicit` | Uses `/index` instead of directory import |

## 🎯 Confidence Levels

| Level | Icon | Certainty | Description |
|-------|------|-----------|-------------|
| HIGH | 🟢 | 95%+ | Exact module match or extension removal - safe to auto-fix |
| MEDIUM | 🟡 | 70-95% | Single partial match - review recommended |
| LOW | 🟠 | <70% | Multiple matches, best guess - manual review needed |
| NONE | ⚫ | 0% | No suggestion available - requires manual fix |

## ⚙️ Command Line Options

### Target Selection

| Option | Description |
|--------|-------------|
| `--tsconfig`, `-c` | Path to tsconfig.json (auto-detected if not specified) |
| `--dir`, `-d` | Directory to scan for imports |
| `--exclude`, `-e` | Additional patterns to exclude (repeatable) |
| `--no-exclude` | Disable default excludes (use with caution!) |

### Fix Options

| Option | Description |
|--------|-------------|
| `--fix-high` | Apply HIGH confidence fixes (safest) |
| `--fix-medium` | Apply MEDIUM confidence fixes |
| `--fix-low` | Apply LOW confidence fixes (risky) |
| `--fix-all` | Apply ALL suggested fixes |
| `--dry-run`, `-n` | Preview changes without modifying files |

### Output Options

| Option | Description |
|--------|-------------|
| `--verbose`, `-v` | Show detailed output and issue details |
| `--quiet`, `-q` | Minimal output - only errors and summary |
| `--export FILE` | Export results to JSON file |
| `--list-modules` | List all discovered modules in the project |

## 📖 Usage Examples

### Scan and Report Only

```bash
python adg-typescript-imports-fixer2.py --dir ./my-vscode-extension/src
```

Output:
```
📋 Using tsconfig: /path/to/tsconfig.json
🔍 Scanning: /path/to/my-vscode-extension/src

════════════════════════════════════════════════════════════════════════════════
  IMPORT SCAN SUMMARY
════════════════════════════════════════════════════════════════════════════════
  📁 Project root:    /path/to/my-vscode-extension
  📂 Scanned:         /path/to/my-vscode-extension/src
  📋 tsconfig.json:   tsconfig.json
     baseUrl:         .
     paths:           3 aliases
────────────────────────────────────────────────────────────────────────────────
  📦 Modules found:   42
  📄 Files scanned:   35
  📥 Total imports:   156

  ✅ Valid imports:   148
  ❌ Invalid imports: 8

  Issues by type:
     🔍 Cannot resolve path: 5
     📎 Has .ts extension: 2
     📄 File not found: 1

  Fixable by confidence:
     🟢 HIGH:   5
     🟡 MEDIUM: 2
     🟠 LOW:    1
     ⚫ NONE:   0
════════════════════════════════════════════════════════════════════════════════
```

### Apply Safe Fixes

```bash
# First, preview what would change
python adg-typescript-imports-fixer2.py --fix-high --dry-run

# If it looks good, apply the fixes
python adg-typescript-imports-fixer2.py --fix-high
```

### Export Results to JSON

```bash
python adg-typescript-imports-fixer2.py --export report.json
```

### CI/CD Integration

```bash
python adg-typescript-imports-fixer2.py --quiet --export report.json
# Exit codes:
#   0 - All imports valid
#   1 - Unfixable issues found
#   2 - Issues found but all fixable
```

## 📂 Supported Import Patterns

```typescript
// ES6 imports
import { foo } from './module'
import foo from './module'
import * as foo from './module'
import './side-effect'
import type { Type } from './types'

// Re-exports
export { foo } from './module'
export * from './module'
export * as utils from './utils'

// Dynamic imports
const module = await import('./module')

// CommonJS
const foo = require('./module')
```

## 🎮 Use Cases

### VS Code Extension Development

When refactoring a VS Code extension, import paths can easily break:

```bash
# Quick scan to find broken imports
python adg-typescript-imports-fixer2.py

# Fix the safe ones
python adg-typescript-imports-fixer2.py --fix-high
```

### Large-Scale Refactoring

After moving files around:

```bash
# See the damage
python adg-typescript-imports-fixer2.py --export before.json

# Apply safe fixes
python adg-typescript-imports-fixer2.py --fix-high

# Apply medium fixes (review output first!)
python adg-typescript-imports-fixer2.py --fix-medium --dry-run
python adg-typescript-imports-fixer2.py --fix-medium

# See what's left
python adg-typescript-imports-fixer2.py --export after.json
```

### Working with tsconfig Aliases

If you use path aliases in `tsconfig.json`:

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@core/*": ["src/core/*"],
      "@utils/*": ["src/utils/*"]
    }
  }
}
```

The tool will:
- ✅ Recognize `@core/module` as an alias import
- ✅ Validate that the target file exists
- ✅ Report if the alias is not defined

## 🔒 What Gets Skipped

- **Package imports**: `lodash`, `@types/node`, `vscode`, etc.
- **Excluded directories**: `node_modules`, `.git`, `dist`, `build`, `out`
- **Configured excludes**: Patterns from `tsconfig.json` exclude array

## 💡 Tips

1. **Always use `--dry-run` first** to preview changes before applying
2. **Make a backup** or commit your changes before running with `--fix-*`
3. **Start with `--fix-high`** - these are the safest fixes
4. **Use `--export`** to create a record of issues for code review
5. **Check your tsconfig.json** - the tool uses it for path resolution

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest features  
- Submit pull requests

## 📄 License

This Source Code Form is subject to the terms of the **Mozilla Public License, v. 2.0**.

If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

---

Made with ❤️ for VS Code extension developers
