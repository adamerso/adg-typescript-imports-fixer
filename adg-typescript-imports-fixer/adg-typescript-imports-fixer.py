#!/usr/bin/env python3
"""
adg-typescript-imports-fixer.py - Intelligent TypeScript/JavaScript Import Path Fixer

A comprehensive tool for analyzing and fixing import paths in TypeScript/JavaScript projects.
Scans directories, validates imports against actual file structure, and provides
multiple confidence levels for automatic fixes.

Author: ADG-Parallels2 Team
Version: 1.0.0
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple
from collections import defaultdict
import json


# ============================================================================
# CONSTANTS
# ============================================================================

VERSION = "1.0.0"

# File extensions to scan
SCANNABLE_EXTENSIONS = {'.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs'}

# Import patterns
IMPORT_PATTERNS = [
    # import { x } from 'path'
    re.compile(r'''import\s+\{[^}]*\}\s+from\s+['"]([^'"]+)['"]'''),
    # import x from 'path'
    re.compile(r'''import\s+\w+\s+from\s+['"]([^'"]+)['"]'''),
    # import * as x from 'path'
    re.compile(r'''import\s+\*\s+as\s+\w+\s+from\s+['"]([^'"]+)['"]'''),
    # import 'path' (side-effect)
    re.compile(r'''import\s+['"]([^'"]+)['"]'''),
    # export { x } from 'path'
    re.compile(r'''export\s+\{[^}]*\}\s+from\s+['"]([^'"]+)['"]'''),
    # export * from 'path'
    re.compile(r'''export\s+\*\s+from\s+['"]([^'"]+)['"]'''),
    # require('path')
    re.compile(r'''require\s*\(\s*['"]([^'"]+)['"]\s*\)'''),
    # import type { x } from 'path'
    re.compile(r'''import\s+type\s+\{[^}]*\}\s+from\s+['"]([^'"]+)['"]'''),
]


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class Confidence(Enum):
    """Confidence level for suggested fixes."""
    HIGH = "high"      # 95%+ certain - exact match found
    MEDIUM = "medium"  # 70-95% - similar name found
    LOW = "low"        # <70% - guessing based on patterns
    NONE = "none"      # No suggestion available


class ImportType(Enum):
    """Type of import path."""
    RELATIVE = "relative"    # ./foo, ../bar
    ABSOLUTE = "absolute"    # /foo/bar
    PACKAGE = "package"      # lodash, @types/node
    ALIAS = "alias"          # @core/utils (tsconfig paths)


@dataclass
class ModuleInfo:
    """Information about a discovered module/file."""
    path: Path                    # Absolute path
    relative_path: Path           # Relative to root
    name: str                     # Module name (without extension)
    extension: str                # File extension
    has_index: bool = False       # True if directory with index.ts
    exports: Set[str] = field(default_factory=set)  # Named exports (optional)


@dataclass
class ImportInfo:
    """Information about an import statement."""
    file_path: Path               # File containing the import
    line_number: int              # Line number (1-indexed)
    line_content: str             # Full line content
    import_path: str              # The imported path
    import_type: ImportType       # Type of import
    is_valid: bool = False        # Whether import resolves
    resolved_to: Optional[Path] = None  # Resolved absolute path
    suggested_fix: Optional[str] = None
    confidence: Confidence = Confidence.NONE
    error_message: Optional[str] = None


@dataclass
class ScanResult:
    """Result of scanning a directory."""
    root: Path
    modules: Dict[str, ModuleInfo]        # name -> ModuleInfo
    imports: List[ImportInfo]              # All found imports
    valid_imports: List[ImportInfo]        # Valid imports
    invalid_imports: List[ImportInfo]      # Invalid imports
    fixable_high: List[ImportInfo]         # HIGH confidence fixes
    fixable_medium: List[ImportInfo]       # MEDIUM confidence fixes
    fixable_low: List[ImportInfo]          # LOW confidence fixes
    unfixable: List[ImportInfo]            # No suggestion


# ============================================================================
# CORE SCANNER CLASS
# ============================================================================

class ImportScanner:
    """
    Scans TypeScript/JavaScript projects for import statements
    and validates them against the actual file structure.
    """
    
    def __init__(
        self,
        root: Path,
        related_to: Optional[Path] = None,
        use_direct_paths: bool = False,
        verbose: bool = False,
        exclude_patterns: Optional[List[str]] = None
    ):
        self.root = root.resolve()
        self.related_to = (related_to or root).resolve()
        self.use_direct_paths = use_direct_paths
        self.verbose = verbose
        self.exclude_patterns = exclude_patterns or [
            'node_modules', '.git', 'dist', 'build', 'out', 
            '__pycache__', '.pytest_cache', 'coverage'
        ]
        
        self.modules: Dict[str, ModuleInfo] = {}
        self.path_to_module: Dict[Path, ModuleInfo] = {}
        self.imports: List[ImportInfo] = []
        
    def log(self, message: str, level: str = "info"):
        """Log message if verbose mode is on."""
        if self.verbose:
            prefix = {"info": "ℹ️ ", "warn": "⚠️ ", "error": "❌", "ok": "✅"}
            print(f"{prefix.get(level, '')} {message}")
    
    def should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded from scanning."""
        for pattern in self.exclude_patterns:
            if pattern in path.parts:
                return True
        return False
    
    def discover_modules(self) -> Dict[str, ModuleInfo]:
        """
        Scan directory tree and discover all modules.
        Returns dict mapping module names to ModuleInfo.
        """
        self.log(f"Scanning for modules in: {self.root}")
        
        for root, dirs, files in os.walk(self.root):
            root_path = Path(root)
            
            # Filter excluded directories
            dirs[:] = [d for d in dirs if not self.should_exclude(root_path / d)]
            
            for file in files:
                file_path = root_path / file
                ext = file_path.suffix.lower()
                
                if ext not in SCANNABLE_EXTENSIONS:
                    continue
                
                relative_path = file_path.relative_to(self.root)
                name = file_path.stem
                
                # Check if this is an index file
                is_index = name == 'index'
                
                module_info = ModuleInfo(
                    path=file_path,
                    relative_path=relative_path,
                    name=name if not is_index else relative_path.parent.name,
                    extension=ext,
                    has_index=is_index
                )
                
                # Register by multiple keys for easy lookup
                # 1. Full relative path without extension (use forward slashes for consistency)
                key_full = str(relative_path.with_suffix('')).replace('\\', '/')
                self.modules[key_full] = module_info
                
                # 2. For index files, also register parent directory
                if is_index:
                    key_dir = str(relative_path.parent).replace('\\', '/')
                    if key_dir != '.':
                        self.modules[key_dir] = module_info
                
                self.path_to_module[file_path] = module_info
        
        self.log(f"Found {len(self.modules)} modules")
        return self.modules
    
    def classify_import(self, import_path: str) -> ImportType:
        """Classify the type of import path."""
        if import_path.startswith('./') or import_path.startswith('../'):
            return ImportType.RELATIVE
        elif import_path.startswith('/'):
            return ImportType.ABSOLUTE
        elif import_path.startswith('@') and '/' in import_path:
            # Could be alias (@core/utils) or scoped package (@types/node)
            # Heuristic: if second part looks like a package, it's a package
            parts = import_path.split('/')
            if parts[0] in ['@types', '@babel', '@jest', '@testing-library']:
                return ImportType.PACKAGE
            return ImportType.ALIAS
        else:
            return ImportType.PACKAGE
    
    def resolve_import(
        self, 
        import_path: str, 
        from_file: Path
    ) -> Tuple[bool, Optional[Path], Optional[str]]:
        """
        Try to resolve an import path to an actual file.
        Returns (is_valid, resolved_path, error_message).
        """
        import_type = self.classify_import(import_path)
        
        # Skip package imports - we can't validate those
        if import_type == ImportType.PACKAGE:
            return (True, None, None)
        
        # Skip alias imports for now (would need tsconfig.json parsing)
        if import_type == ImportType.ALIAS:
            return (True, None, "Alias import - needs tsconfig.json")
        
        # Resolve relative import
        from_dir = from_file.parent
        
        # Handle the import path
        target = (from_dir / import_path).resolve()
        
        # Try different extensions
        extensions_to_try = ['', '.ts', '.tsx', '.js', '.jsx', '/index.ts', '/index.tsx', '/index.js']
        
        for ext in extensions_to_try:
            candidate = Path(str(target) + ext)
            if candidate.exists() and candidate.is_file():
                return (True, candidate, None)
        
        return (False, None, f"Cannot resolve: {import_path}")
    
    def suggest_fix(
        self, 
        import_info: ImportInfo
    ) -> Tuple[Optional[str], Confidence]:
        """
        Suggest a fix for an invalid import.
        Returns (suggested_path, confidence).
        """
        original = import_info.import_path
        from_file = import_info.file_path
        from_dir = from_file.parent
        
        # Extract the target module name from the import
        # e.g., "../../types/shared" -> "types/shared"
        # e.g., "../src_v2/types" -> "types"
        
        # Remove leading ./ and ../
        clean_path = re.sub(r'^(\.\./|\./)+', '', original)
        # Remove src_v2/ if present (common mistake)
        clean_path = re.sub(r'^src_v2/', '', clean_path)
        
        # SPECIAL HANDLING: If path contains 'types/', preserve it!
        # This prevents matching 'types/orchestration' to 'core/orchestration'
        is_types_import = clean_path.startswith('types') if clean_path else False
        
        # Helper function to calculate relative path
        def calc_relative(target_module: ModuleInfo) -> Optional[str]:
            target_path = target_module.path
            if target_module.has_index:
                target_path = target_module.path.parent
            try:
                rel_path = os.path.relpath(target_path, from_dir)
                rel_path = rel_path.replace('\\', '/')
                if not rel_path.startswith('.'):
                    rel_path = './' + rel_path
                rel_path = re.sub(r'\.tsx?$', '', rel_path)
                rel_path = re.sub(r'/index$', '', rel_path)
                return rel_path
            except ValueError:
                return None
        
        # Try to find this module in our discovered modules
        if clean_path in self.modules:
            module = self.modules[clean_path]
            rel_path = calc_relative(module)
            if rel_path:
                return (rel_path, Confidence.HIGH)
        
        # For types imports, try types/* patterns
        if is_types_import:
            # clean_path is like "types/orchestration" or "types"
            # Check if it exists
            for key, module in self.modules.items():
                if key == clean_path or key == clean_path.rstrip('/'):
                    rel_path = calc_relative(module)
                    if rel_path:
                        return (rel_path, Confidence.HIGH)
            
            # If not found by exact match, it means the path depth is wrong
            # but the target module exists - search for it
            parts = clean_path.split('/')
            if len(parts) >= 1:
                # Try to find types/X where X is last part
                search_key = clean_path
                if search_key in self.modules:
                    module = self.modules[search_key]
                    rel_path = calc_relative(module)
                    if rel_path:
                        return (rel_path, Confidence.HIGH)
        
        # Try partial match - find modules containing the last part
        last_part = clean_path.split('/')[-1] if clean_path else ''
        candidates = []
        
        for key, module in self.modules.items():
            # If this is a types import, only match in types/ directory
            if is_types_import:
                if not key.startswith('types/') and key != 'types':
                    continue
            
            if key.endswith(last_part) or key.endswith(f"/{last_part}"):
                candidates.append((key, module))
            # Also try exact suffix match for paths like "types/shared"
            elif clean_path and key == clean_path:
                candidates.append((key, module))
        
        if len(candidates) == 1:
            # Single match - high confidence for types
            module = candidates[0][1]
            rel_path = calc_relative(module)
            if rel_path:
                conf = Confidence.HIGH if is_types_import else Confidence.MEDIUM
                return (rel_path, conf)
        
        elif len(candidates) > 1:
            # Multiple matches - prefer types/ path if is_types_import
            if is_types_import:
                types_candidates = [(k, m) for k, m in candidates if k.startswith('types')]
                if len(types_candidates) == 1:
                    candidates = types_candidates
                    module = candidates[0][1]
                    rel_path = calc_relative(module)
                    if rel_path:
                        return (rel_path, Confidence.HIGH)
            
            # Pick shortest path
            candidates.sort(key=lambda x: len(x[0]))
            module = candidates[0][1]
            rel_path = calc_relative(module)
            if rel_path:
                return (rel_path, Confidence.LOW)
        
        return (None, Confidence.NONE)
    
    def scan_file(self, file_path: Path) -> List[ImportInfo]:
        """Scan a single file for imports."""
        imports = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            self.log(f"Cannot read {file_path}: {e}", "error")
            return imports
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*'):
                continue
            
            for pattern in IMPORT_PATTERNS:
                for match in pattern.finditer(line):
                    import_path = match.group(1)
                    import_type = self.classify_import(import_path)
                    
                    # Validate the import
                    is_valid, resolved, error = self.resolve_import(import_path, file_path)
                    
                    import_info = ImportInfo(
                        file_path=file_path,
                        line_number=line_num,
                        line_content=line.strip(),
                        import_path=import_path,
                        import_type=import_type,
                        is_valid=is_valid,
                        resolved_to=resolved,
                        error_message=error
                    )
                    
                    # Suggest fix if invalid
                    if not is_valid and import_type == ImportType.RELATIVE:
                        fix, confidence = self.suggest_fix(import_info)
                        import_info.suggested_fix = fix
                        import_info.confidence = confidence
                    
                    imports.append(import_info)
        
        return imports
    
    def scan_all(self) -> ScanResult:
        """Scan all files in the root directory."""
        self.discover_modules()
        
        all_imports = []
        
        for root, dirs, files in os.walk(self.root):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not self.should_exclude(root_path / d)]
            
            for file in files:
                file_path = root_path / file
                if file_path.suffix.lower() in SCANNABLE_EXTENSIONS:
                    file_imports = self.scan_file(file_path)
                    all_imports.extend(file_imports)
        
        self.imports = all_imports
        
        # Categorize imports
        valid = [i for i in all_imports if i.is_valid]
        invalid = [i for i in all_imports if not i.is_valid]
        
        fixable_high = [i for i in invalid if i.confidence == Confidence.HIGH]
        fixable_medium = [i for i in invalid if i.confidence == Confidence.MEDIUM]
        fixable_low = [i for i in invalid if i.confidence == Confidence.LOW]
        unfixable = [i for i in invalid if i.confidence == Confidence.NONE]
        
        return ScanResult(
            root=self.root,
            modules=self.modules,
            imports=all_imports,
            valid_imports=valid,
            invalid_imports=invalid,
            fixable_high=fixable_high,
            fixable_medium=fixable_medium,
            fixable_low=fixable_low,
            unfixable=unfixable
        )


# ============================================================================
# FIXER CLASS
# ============================================================================

class ImportFixer:
    """Applies fixes to import statements."""
    
    def __init__(
        self, 
        dry_run: bool = True, 
        verbose: bool = False,
        scan_root: Optional[Path] = None
    ):
        self.dry_run = dry_run
        self.verbose = verbose
        self.scan_root = scan_root or Path.cwd()
        self.workdir = Path.cwd()
        self.changes_made: Dict[Path, List[Tuple[int, str, str]]] = defaultdict(list)
    
    def log(self, message: str):
        if self.verbose:
            print(message)
    
    def apply_fix(self, import_info: ImportInfo) -> bool:
        """Apply a single fix. Returns True if successful."""
        if not import_info.suggested_fix:
            return False
        
        file_path = import_info.file_path
        old_import = import_info.import_path
        new_import = import_info.suggested_fix
        
        # Record the change
        self.changes_made[file_path].append((
            import_info.line_number,
            old_import,
            new_import
        ))
        
        if self.dry_run:
            return True
        
        try:
            content = file_path.read_text(encoding='utf-8')
            # Replace the specific import (be careful with quotes)
            old_patterns = [f"'{old_import}'", f'"{old_import}"']
            new_patterns = [f"'{new_import}'", f'"{new_import}"']
            
            for old, new in zip(old_patterns, new_patterns):
                content = content.replace(old, new)
            
            file_path.write_text(content, encoding='utf-8')
            return True
        except Exception as e:
            self.log(f"Error fixing {file_path}: {e}")
            return False
    
    def apply_fixes(
        self, 
        imports: List[ImportInfo],
        confidence_levels: Set[Confidence]
    ) -> Tuple[int, int]:
        """
        Apply fixes for imports matching confidence levels.
        Returns (successful, failed).
        """
        successful = 0
        failed = 0
        
        for imp in imports:
            if imp.confidence in confidence_levels and imp.suggested_fix:
                if self.apply_fix(imp):
                    successful += 1
                else:
                    failed += 1
        
        return (successful, failed)
    
    def print_changes(self):
        """Print summary of changes with full path visibility."""
        if not self.changes_made:
            print("No changes to apply.")
            return
        
        print("\n" + "=" * 70)
        print("CHANGES " + ("(DRY-RUN)" if self.dry_run else "(APPLIED)"))
        print("=" * 70)
        print(f"📂 Workdir:   {self.workdir}")
        print(f"📁 Scan root: {self.scan_root}")
        print("-" * 70)
        
        for file_path, changes in sorted(self.changes_made.items()):
            # Show both paths for clarity
            try:
                rel_to_workdir = file_path.relative_to(self.workdir)
            except ValueError:
                rel_to_workdir = file_path
            
            try:
                rel_to_root = file_path.relative_to(self.scan_root)
            except ValueError:
                rel_to_root = file_path
            
            print(f"\n📄 {rel_to_root}")
            print(f"   (workdir: {rel_to_workdir})")
            
            for line_num, old, new in changes:
                print(f"   L{line_num}: '{old}' → '{new}'")
        
        total = sum(len(c) for c in self.changes_made.values())
        print(f"\n{'Would change' if self.dry_run else 'Changed'}: {total} imports in {len(self.changes_made)} files")


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generates reports from scan results."""
    
    @staticmethod
    def print_summary(result: ScanResult):
        """Print a summary of the scan results."""
        print("\n" + "=" * 60)
        print("IMPORT SCAN SUMMARY")
        print("=" * 60)
        print(f"📁 Root: {result.root}")
        print(f"📦 Modules found: {len(result.modules)}")
        print(f"📥 Total imports: {len(result.imports)}")
        print()
        print(f"✅ Valid imports:   {len(result.valid_imports)}")
        print(f"❌ Invalid imports: {len(result.invalid_imports)}")
        print()
        print("Fixable by confidence:")
        print(f"  🟢 HIGH:   {len(result.fixable_high)}")
        print(f"  🟡 MEDIUM: {len(result.fixable_medium)}")
        print(f"  🟠 LOW:    {len(result.fixable_low)}")
        print(f"  ⚫ NONE:   {len(result.unfixable)}")
    
    @staticmethod
    def print_invalid_imports(result: ScanResult, show_all: bool = False):
        """Print details of invalid imports."""
        if not result.invalid_imports:
            print("\n✅ No invalid imports found!")
            return
        
        print("\n" + "=" * 60)
        print("INVALID IMPORTS")
        print("=" * 60)
        
        # Group by file
        by_file: Dict[Path, List[ImportInfo]] = defaultdict(list)
        for imp in result.invalid_imports:
            by_file[imp.file_path].append(imp)
        
        for file_path, imports in sorted(by_file.items()):
            rel_path = file_path.relative_to(result.root)
            print(f"\n📄 {rel_path}:")
            
            for imp in imports:
                conf_icon = {
                    Confidence.HIGH: "🟢",
                    Confidence.MEDIUM: "🟡",
                    Confidence.LOW: "🟠",
                    Confidence.NONE: "⚫"
                }[imp.confidence]
                
                print(f"   L{imp.line_number}: {conf_icon} '{imp.import_path}'")
                if imp.suggested_fix:
                    print(f"         → '{imp.suggested_fix}'")
                if imp.error_message and show_all:
                    print(f"         ⚠️  {imp.error_message}")
    
    @staticmethod
    def export_json(result: ScanResult, output_path: Path):
        """Export results to JSON."""
        data = {
            "root": str(result.root),
            "summary": {
                "total_modules": len(result.modules),
                "total_imports": len(result.imports),
                "valid_imports": len(result.valid_imports),
                "invalid_imports": len(result.invalid_imports),
                "fixable_high": len(result.fixable_high),
                "fixable_medium": len(result.fixable_medium),
                "fixable_low": len(result.fixable_low),
                "unfixable": len(result.unfixable)
            },
            "invalid_imports": [
                {
                    "file": str(imp.file_path.relative_to(result.root)),
                    "line": imp.line_number,
                    "import": imp.import_path,
                    "suggested": imp.suggested_fix,
                    "confidence": imp.confidence.value
                }
                for imp in result.invalid_imports
            ]
        }
        
        output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        print(f"\n📊 Report exported to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with rich help."""
    
    parser = argparse.ArgumentParser(
        prog='adg-typescript-imports-fixer',
        description='''
╔══════════════════════════════════════════════════════════════════════════════╗
║              ADG-TYPESCRIPT-IMPORTS-FIXER - Import Path Fixer                ║
║                                                                              ║
║  Scans TypeScript/JavaScript projects for broken import paths and suggests  ║
║  fixes with different confidence levels. Can automatically apply fixes.     ║
╚══════════════════════════════════════════════════════════════════════════════╝

EXAMPLES:
  # Scan and show report (no changes)
  python adg-typescript-imports-fixer.py ./src

  # Dry-run: show what HIGH confidence fixes would change
  python adg-typescript-imports-fixer.py ./src --fix-high --dry-run

  # Apply HIGH confidence fixes
  python adg-typescript-imports-fixer.py ./src --fix-high

  # Apply HIGH and MEDIUM confidence fixes
  python adg-typescript-imports-fixer.py ./src --fix-high --fix-medium

  # Aggressive: apply all suggested fixes
  python adg-typescript-imports-fixer.py ./src --fix-all

  # Scan with custom base directory
  python adg-typescript-imports-fixer.py ./src --related-to ./project-root

  # Export report to JSON
  python adg-typescript-imports-fixer.py ./src --export report.json
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
CONFIDENCE LEVELS:
  🟢 HIGH   - Exact module match found. 95%+ certain the fix is correct.
  🟡 MEDIUM - Single partial match found. 70-95% certain.
  🟠 LOW    - Multiple matches or pattern-based guess. <70% certain.
  ⚫ NONE   - No suggestion available. Manual fix required.

NOTES:
  - Package imports (lodash, @types/node) are skipped - cannot validate.
  - Alias imports (@core/utils) are skipped - would need tsconfig.json.
  - Always use --dry-run first to preview changes!
  - Creates backup recommended before --fix-* operations.
'''
    )
    
    # Positional
    parser.add_argument(
        'directory',
        type=Path,
        nargs='?',
        default=Path('.'),
        help='Directory to scan (default: current directory)'
    )
    
    # Scan options
    scan_group = parser.add_argument_group('Scan Options')
    scan_group.add_argument(
        '--related-to', '-r',
        type=Path,
        metavar='DIR',
        help='Base directory for resolving paths (default: same as directory)'
    )
    scan_group.add_argument(
        '--exclude', '-e',
        action='append',
        metavar='PATTERN',
        help='Additional directory patterns to exclude (can be used multiple times)'
    )
    
    # Fix options
    fix_group = parser.add_argument_group('Fix Options')
    fix_group.add_argument(
        '--fix-high',
        action='store_true',
        help='Apply HIGH confidence fixes (safest)'
    )
    fix_group.add_argument(
        '--fix-medium',
        action='store_true',
        help='Apply MEDIUM confidence fixes'
    )
    fix_group.add_argument(
        '--fix-low',
        action='store_true',
        help='Apply LOW confidence fixes (risky)'
    )
    fix_group.add_argument(
        '--fix-all',
        action='store_true',
        help='Apply ALL suggested fixes (equivalent to --fix-high --fix-medium --fix-low)'
    )
    fix_group.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be changed without making changes'
    )
    fix_group.add_argument(
        '--direct-paths',
        action='store_true',
        help='Use absolute paths instead of relative (not recommended)'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output during scanning'
    )
    output_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only show errors and summary'
    )
    output_group.add_argument(
        '--export',
        type=Path,
        metavar='FILE',
        help='Export results to JSON file'
    )
    output_group.add_argument(
        '--show-valid',
        action='store_true',
        help='Also show valid imports in report'
    )
    
    # Other
    parser.add_argument(
        '--version', '-V',
        action='version',
        version=f'%(prog)s {VERSION}'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Validate directory
    directory = args.directory.resolve()
    if not directory.exists():
        print(f"❌ Error: Directory does not exist: {directory}")
        sys.exit(1)
    if not directory.is_dir():
        print(f"❌ Error: Not a directory: {directory}")
        sys.exit(1)
    
    # Build exclude list
    exclude = None
    if args.exclude:
        exclude = args.exclude
    
    # Create scanner
    scanner = ImportScanner(
        root=directory,
        related_to=args.related_to,
        use_direct_paths=args.direct_paths,
        verbose=args.verbose,
        exclude_patterns=exclude
    )
    
    # Scan
    if not args.quiet:
        print(f"🔍 Scanning: {directory}")
    
    result = scanner.scan_all()
    
    # Print report
    if not args.quiet:
        ReportGenerator.print_summary(result)
        ReportGenerator.print_invalid_imports(result, show_all=args.verbose)
    
    # Export if requested
    if args.export:
        ReportGenerator.export_json(result, args.export)
    
    # Apply fixes if requested
    should_fix = args.fix_high or args.fix_medium or args.fix_low or args.fix_all
    
    if should_fix:
        # Determine which confidence levels to fix
        levels: Set[Confidence] = set()
        if args.fix_high or args.fix_all:
            levels.add(Confidence.HIGH)
        if args.fix_medium or args.fix_all:
            levels.add(Confidence.MEDIUM)
        if args.fix_low or args.fix_all:
            levels.add(Confidence.LOW)
        
        # If no --dry-run and making changes, warn
        if not args.dry_run and levels:
            level_names = ', '.join(l.value.upper() for l in levels)
            print(f"\n⚠️  About to apply {level_names} confidence fixes.")
            print("   Make sure you have a backup!")
            response = input("   Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
        
        # Create fixer
        fixer = ImportFixer(
            dry_run=args.dry_run,
            verbose=args.verbose,
            scan_root=directory
        )
        
        # Apply fixes
        imports_to_fix = [
            i for i in result.invalid_imports 
            if i.confidence in levels and i.suggested_fix
        ]
        
        if imports_to_fix:
            for imp in imports_to_fix:
                fixer.apply_fix(imp)
            
            fixer.print_changes()
        else:
            print("\nNo fixes to apply for selected confidence levels.")
    
    # Exit with error code if there are unfixable imports
    if result.unfixable:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
