#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 ADG-Parallels2 Team
"""
adg-typescript-imports-fixer v2.0 - TypeScript/JavaScript Import Path Analyzer & Fixer

A comprehensive tool for analyzing and fixing import paths in TypeScript/JavaScript 
projects. Works relative to tsconfig.json, validates every import in every file,
and provides detailed diagnostics.

Features:
- Reads and respects tsconfig.json (paths, baseUrl, include, exclude)
- Validates EVERY import statement in EVERY file
- Detects missing files (path structure exists but file doesn't)
- Detects leftover .ts/.tsx extensions in imports (TypeScript anti-pattern)
- Tests paths from the importing file's perspective (correct relative resolution)
- Multiple confidence levels for auto-fixes
- Dry-run mode for safe previewing
- JSON export for CI/CD integration

Author: ADG-Parallels2 Team
License: Mozilla Public License 2.0
Version: 2.0.0
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple, Any
from collections import defaultdict


# ============================================================================
# CONSTANTS
# ============================================================================

VERSION = "2.0.0"
LICENSE = "Mozilla Public License 2.0"

# File extensions to scan for imports
SCANNABLE_EXTENSIONS = {'.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs'}

# Extensions to try when resolving imports (order matters!)
RESOLVE_EXTENSIONS = [
    '',           # exact match first
    '.ts', 
    '.tsx', 
    '.js', 
    '.jsx',
    '.mjs',
    '.cjs',
    '/index.ts', 
    '/index.tsx', 
    '/index.js',
    '/index.jsx'
]

# Import/export patterns - capture: (prefix)(path)(suffix)
IMPORT_PATTERNS = [
    # import type { x } from 'path' (must be before regular import)
    re.compile(r'''(import\s+type\s+\{[^}]*\}\s+from\s+['"])([^'"]+)(['"])'''),
    # import type x from 'path'
    re.compile(r'''(import\s+type\s+[\w]+\s+from\s+['"])([^'"]+)(['"])'''),
    # import { x } from 'path'
    re.compile(r'''(import\s+\{[^}]*\}\s+from\s+['"])([^'"]+)(['"])'''),
    # import x from 'path'
    re.compile(r'''(import\s+[\w]+\s+from\s+['"])([^'"]+)(['"])'''),
    # import * as x from 'path'
    re.compile(r'''(import\s+\*\s+as\s+\w+\s+from\s+['"])([^'"]+)(['"])'''),
    # import 'path' (side-effect only)
    re.compile(r'''(import\s+['"])([^'"]+)(['"])(?!\s+from)'''),
    # export { x } from 'path'
    re.compile(r'''(export\s+\{[^}]*\}\s+from\s+['"])([^'"]+)(['"])'''),
    # export * from 'path'
    re.compile(r'''(export\s+\*\s+from\s+['"])([^'"]+)(['"])'''),
    # export * as x from 'path'
    re.compile(r'''(export\s+\*\s+as\s+\w+\s+from\s+['"])([^'"]+)(['"])'''),
    # require('path')
    re.compile(r'''(require\s*\(\s*['"])([^'"]+)(['"]\s*\))'''),
    # dynamic import('path')
    re.compile(r'''(import\s*\(\s*['"])([^'"]+)(['"]\s*\))'''),
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
    ABSOLUTE = "absolute"    # /foo/bar (rare in TS)
    PACKAGE = "package"      # lodash, @types/node
    ALIAS = "alias"          # @core/utils (tsconfig paths)


class IssueType(Enum):
    """Type of import issue detected."""
    CANNOT_RESOLVE = "cannot_resolve"       # Path doesn't resolve to any file
    MISSING_FILE = "missing_file"           # Directory exists but file doesn't
    HAS_TS_EXTENSION = "has_ts_extension"   # Import has .ts/.tsx extension (anti-pattern)
    HAS_JS_EXTENSION = "has_js_extension"   # Import has .js/.jsx extension (may be intentional)
    WRONG_DEPTH = "wrong_depth"             # Too many ../ (escapes project)
    ALIAS_NOT_FOUND = "alias_not_found"     # @alias not in tsconfig paths
    INDEX_EXPLICIT = "index_explicit"       # Imports /index explicitly (could use directory)


@dataclass
class TsConfig:
    """Parsed tsconfig.json configuration."""
    config_path: Path
    base_url: Optional[Path] = None
    paths: Dict[str, List[str]] = field(default_factory=dict)
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    root_dir: Optional[Path] = None
    out_dir: Optional[Path] = None
    
    @property
    def project_root(self) -> Path:
        """Return the project root (directory containing tsconfig)."""
        return self.config_path.parent


@dataclass
class ModuleInfo:
    """Information about a discovered module/file."""
    absolute_path: Path           # Absolute path to file
    relative_to_root: str         # Relative to project root (forward slashes)
    name: str                     # Module name (without extension)
    extension: str                # File extension
    is_index: bool = False        # True if this is an index.ts file
    directory: Optional[str] = None  # Parent directory path for index files


@dataclass 
class ImportStatement:
    """Information about a single import statement."""
    file_path: Path               # File containing the import
    line_number: int              # Line number (1-indexed)
    column: int                   # Column where import path starts
    full_line: str                # Full line content
    import_path: str              # The imported path as written
    import_type: ImportType       # Classification of import
    
    # Validation results
    is_valid: bool = False
    resolved_path: Optional[Path] = None
    issues: List[IssueType] = field(default_factory=list)
    issue_details: Optional[str] = None
    
    # Fix suggestion
    suggested_fix: Optional[str] = None
    confidence: Confidence = Confidence.NONE
    fix_explanation: Optional[str] = None


@dataclass
class ScanResult:
    """Complete scan results."""
    project_root: Path
    scan_directory: Path
    tsconfig: Optional[TsConfig]
    modules: Dict[str, ModuleInfo]
    all_imports: List[ImportStatement]
    valid_imports: List[ImportStatement]
    invalid_imports: List[ImportStatement]
    
    # Categorized by issue type
    by_issue: Dict[IssueType, List[ImportStatement]] = field(default_factory=dict)
    
    # Categorized by confidence
    fixable_high: List[ImportStatement] = field(default_factory=list)
    fixable_medium: List[ImportStatement] = field(default_factory=list)
    fixable_low: List[ImportStatement] = field(default_factory=list)
    unfixable: List[ImportStatement] = field(default_factory=list)
    
    # Stats
    files_scanned: int = 0


# ============================================================================
# TSCONFIG PARSER
# ============================================================================

class TsConfigParser:
    """Parses tsconfig.json and handles JSONC (JSON with comments)."""
    
    @staticmethod
    def find_tsconfig(start_path: Path) -> Optional[Path]:
        """Find tsconfig.json by walking up the directory tree."""
        current = start_path.resolve()
        if current.is_file():
            current = current.parent
        
        while current != current.parent:
            tsconfig_path = current / 'tsconfig.json'
            if tsconfig_path.exists():
                return tsconfig_path
            current = current.parent
        return None
    
    @staticmethod
    def _strip_jsonc_comments(content: str) -> str:
        """Remove comments from JSONC content while preserving strings."""
        result = []
        i = 0
        in_string = False
        
        while i < len(content):
            char = content[i]
            
            # Handle string boundaries (but not escaped quotes)
            if char == '"' and (i == 0 or content[i-1] != '\\'):
                in_string = not in_string
                result.append(char)
                i += 1
            # Single-line comment (only outside strings)
            elif not in_string and i < len(content) - 1 and content[i:i+2] == '//':
                # Skip until end of line
                while i < len(content) and content[i] != '\n':
                    i += 1
            # Multi-line comment (only outside strings)
            elif not in_string and i < len(content) - 1 and content[i:i+2] == '/*':
                # Skip until */
                i += 2
                while i < len(content) - 1 and content[i:i+2] != '*/':
                    i += 1
                i += 2  # Skip closing */
            else:
                result.append(char)
                i += 1
        
        return ''.join(result)
    
    @staticmethod
    def parse(config_path: Path) -> TsConfig:
        """Parse tsconfig.json file (with JSONC support)."""
        if not config_path.exists():
            raise FileNotFoundError(f"tsconfig.json not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Strip comments
        content = TsConfigParser._strip_jsonc_comments(content)
        
        # Remove trailing commas (JSONC feature)
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_path}: {e}")
        
        config_dir = config_path.parent.resolve()
        compiler_options = data.get('compilerOptions', {})
        
        # Parse baseUrl (relative to config file)
        base_url = None
        if 'baseUrl' in compiler_options:
            base_url = (config_dir / compiler_options['baseUrl']).resolve()
        
        # Parse paths (aliases)
        paths = compiler_options.get('paths', {})
        
        # Parse rootDir
        root_dir = None
        if 'rootDir' in compiler_options:
            root_dir = (config_dir / compiler_options['rootDir']).resolve()
        
        # Parse outDir
        out_dir = None
        if 'outDir' in compiler_options:
            out_dir = (config_dir / compiler_options['outDir']).resolve()
        
        return TsConfig(
            config_path=config_path,
            base_url=base_url or config_dir,
            paths=paths,
            include=data.get('include', ['**/*']),
            exclude=data.get('exclude', ['node_modules']),
            root_dir=root_dir,
            out_dir=out_dir
        )


# ============================================================================
# PATH RESOLVER
# ============================================================================

class PathResolver:
    """Resolves import paths to actual files - from the importing file's perspective."""
    
    def __init__(
        self, 
        project_root: Path,
        tsconfig: Optional[TsConfig] = None,
        modules: Optional[Dict[str, ModuleInfo]] = None
    ):
        self.project_root = project_root.resolve()
        self.tsconfig = tsconfig
        self.modules = modules or {}
        self.base_url = (tsconfig.base_url if tsconfig else project_root).resolve()
        
        # Pre-compile alias patterns for efficiency
        self._alias_patterns = []
        if tsconfig and tsconfig.paths:
            for alias, targets in tsconfig.paths.items():
                # Convert glob pattern to regex: @app/* -> @app/(.*)
                if alias.endswith('/*'):
                    pattern = re.escape(alias[:-2]) + r'/(.+)'
                    base = alias[:-2]
                else:
                    pattern = re.escape(alias) + r'$'
                    base = alias
                self._alias_patterns.append((re.compile(pattern), base, targets))
    
    def classify_import(self, import_path: str) -> ImportType:
        """Classify the type of import path."""
        if import_path.startswith('./') or import_path.startswith('../'):
            return ImportType.RELATIVE
        elif import_path.startswith('/'):
            return ImportType.ABSOLUTE
        elif import_path.startswith('@'):
            # Check if it matches any tsconfig alias
            for pattern, base, _ in self._alias_patterns:
                if pattern.match(import_path) or import_path == base:
                    return ImportType.ALIAS
            # Common scoped packages
            scoped_packages = [
                '@types/', '@babel/', '@jest/', '@testing-library/', 
                '@angular/', '@vue/', '@react/', '@nestjs/', '@emotion/',
                '@mui/', '@chakra-ui/', '@radix-ui/', '@headlessui/',
                '@tanstack/', '@trpc/', '@prisma/', '@supabase/'
            ]
            for pkg in scoped_packages:
                if import_path.startswith(pkg):
                    return ImportType.PACKAGE
            # If not a known package and starts with @, might be alias
            return ImportType.ALIAS
        else:
            return ImportType.PACKAGE
    
    def resolve_alias(self, import_path: str) -> Optional[str]:
        """Resolve a tsconfig path alias to actual relative path."""
        for pattern, base, targets in self._alias_patterns:
            match = pattern.match(import_path)
            if match:
                # Get the captured part after the alias
                rest = match.group(1) if match.lastindex else ''
                
                # Try each target mapping
                for target in targets:
                    if target.endswith('/*'):
                        resolved = target[:-2] + '/' + rest if rest else target[:-2]
                    else:
                        resolved = target
                    return resolved
            
            # Exact match (no wildcard)
            if import_path == base:
                for target in targets:
                    return target.rstrip('/*')
        
        return None
    
    def resolve_import(
        self, 
        import_path: str, 
        from_file: Path
    ) -> Tuple[bool, Optional[Path], List[IssueType], Optional[str]]:
        """
        Resolve an import path from the perspective of from_file.
        
        Args:
            import_path: The import path as written in the source
            from_file: Absolute path to the file containing the import
            
        Returns: (is_valid, resolved_path, issues, details)
        """
        issues = []
        details = None
        
        import_type = self.classify_import(import_path)
        
        # Check for TypeScript extension in import (anti-pattern!)
        if import_path.endswith('.ts') or import_path.endswith('.tsx'):
            issues.append(IssueType.HAS_TS_EXTENSION)
        elif import_path.endswith('.js') or import_path.endswith('.jsx'):
            issues.append(IssueType.HAS_JS_EXTENSION)
        
        # Check for explicit /index (could use directory import)
        if import_path.endswith('/index') or '/index.' in import_path:
            issues.append(IssueType.INDEX_EXPLICIT)
        
        # Skip package imports - can't validate without node_modules analysis
        if import_type == ImportType.PACKAGE:
            return (True, None, issues, "Package import - assumed valid")
        
        # Resolve the target path
        target: Optional[Path] = None
        
        if import_type == ImportType.ALIAS:
            resolved_alias = self.resolve_alias(import_path)
            if resolved_alias is None:
                issues.append(IssueType.ALIAS_NOT_FOUND)
                return (False, None, issues, f"Alias '{import_path.split('/')[0]}' not found in tsconfig.paths")
            # Resolve alias relative to baseUrl
            target = (self.base_url / resolved_alias).resolve()
        
        elif import_type == ImportType.RELATIVE:
            # CRITICAL: Resolve relative to the IMPORTING FILE's directory
            from_dir = from_file.parent.resolve()
            target = (from_dir / import_path).resolve()
        
        elif import_type == ImportType.ABSOLUTE:
            target = Path(import_path).resolve()
        
        if target is None:
            issues.append(IssueType.CANNOT_RESOLVE)
            return (False, None, issues, "Could not determine target path")
        
        # Check if path escapes project root (too many ../)
        try:
            target.relative_to(self.project_root)
        except ValueError:
            issues.append(IssueType.WRONG_DEPTH)
            details = f"Path escapes project root: {target}"
            issues.append(IssueType.CANNOT_RESOLVE)
            return (False, None, issues, details)
        
        # Try to find the actual file
        resolved = self._find_file(target)
        
        if resolved:
            # File found - only extension issues remain
            return (len(issues) == 0, resolved, issues, None)
        
        # File not found - determine why
        issues.append(IssueType.CANNOT_RESOLVE)
        
        # Check if directory exists but file doesn't (common after refactoring)
        target_dir = target.parent
        target_name = target.name
        
        if target_dir.exists():
            # Directory exists - file was likely moved/deleted
            issues.append(IssueType.MISSING_FILE)
            
            # List what's in the directory as hint
            try:
                existing = [f.name for f in target_dir.iterdir() if f.is_file()]
                if existing:
                    similar = [f for f in existing if target_name.lower() in f.lower()]
                    if similar:
                        details = f"File not found in {target_dir.name}/. Similar: {', '.join(similar[:3])}"
                    else:
                        details = f"File '{target_name}' not found in {target_dir.name}/"
                else:
                    details = f"Directory {target_dir.name}/ is empty"
            except PermissionError:
                details = f"Cannot access directory: {target_dir}"
        else:
            details = f"Directory does not exist: {target_dir}"
        
        return (False, None, issues, details)
    
    def _find_file(self, target: Path) -> Optional[Path]:
        """Try to find actual file with various extension combinations."""
        target = target.resolve()
        
        for ext in RESOLVE_EXTENSIONS:
            if ext.startswith('/'):
                # It's an index variant
                candidate = target / ext[1:]
            else:
                candidate = Path(str(target) + ext)
            
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()
            except (OSError, ValueError):
                continue
        
        return None
    
    def calculate_relative_path(
        self,
        target_module: ModuleInfo,
        from_file: Path
    ) -> Optional[str]:
        """
        Calculate correct relative import path from from_file to target_module.
        
        This is the FIX generator - it creates the correct path.
        """
        try:
            from_dir = from_file.parent.resolve()
            target = target_module.absolute_path.resolve()
            
            # If it's an index file, import the directory instead
            if target_module.is_index:
                target = target.parent
            
            # Calculate relative path
            rel_path = os.path.relpath(target, from_dir)
            rel_path = rel_path.replace('\\', '/')
            
            # Ensure it starts with ./ or ../
            if not rel_path.startswith('.'):
                rel_path = './' + rel_path
            
            # Remove .ts/.tsx extension (TypeScript convention)
            rel_path = re.sub(r'\.(ts|tsx|js|jsx|mjs|cjs)$', '', rel_path)
            
            # Remove trailing /index (use directory import)
            rel_path = re.sub(r'/index$', '', rel_path)
            
            return rel_path
        except (ValueError, OSError):
            return None


# ============================================================================
# IMPORT SCANNER
# ============================================================================

class ImportScanner:
    """Scans TypeScript/JavaScript files for import statements and validates them."""
    
    def __init__(
        self,
        project_root: Path,
        scan_directory: Optional[Path] = None,
        tsconfig: Optional[TsConfig] = None,
        verbose: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        no_exclude: bool = False
    ):
        self.project_root = project_root.resolve()
        self.scan_directory = (scan_directory or project_root).resolve()
        self.tsconfig = tsconfig
        self.verbose = verbose
        
        # Build exclude set
        if no_exclude:
            self.exclude_patterns = set(exclude_patterns or [])
        else:
            self.exclude_patterns = set([
                'node_modules', '.git', 'dist', 'build', 'out',
                '__pycache__', '.pytest_cache', 'coverage', '.next',
                '.nuxt', '.output', '.vercel', '.netlify'
            ])
            if tsconfig and tsconfig.exclude:
                self.exclude_patterns.update(tsconfig.exclude)
            if exclude_patterns:
                self.exclude_patterns.update(exclude_patterns)
        
        self.modules: Dict[str, ModuleInfo] = {}
        self.resolver: Optional[PathResolver] = None
        self._files_scanned = 0
    
    def log(self, message: str, level: str = "info"):
        """Log if verbose mode is enabled."""
        if self.verbose:
            prefix = {"info": "ℹ️ ", "warn": "⚠️ ", "error": "❌", "ok": "✅", "debug": "🔍"}
            print(f"{prefix.get(level, '')} {message}")
    
    def should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded from scanning."""
        path_parts = path.parts
        path_str = str(path)
        
        for pattern in self.exclude_patterns:
            if pattern in path_parts:
                return True
            # Also check as substring for patterns like "*.d.ts"
            if '*' in pattern:
                import fnmatch
                if fnmatch.fnmatch(path.name, pattern):
                    return True
        return False
    
    def discover_modules(self) -> Dict[str, ModuleInfo]:
        """
        Discover all TypeScript/JavaScript modules in the PROJECT ROOT.
        This builds the registry of all possible import targets.
        """
        self.log(f"Discovering modules in: {self.project_root}")
        
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            
            # Filter excluded directories in-place
            dirs[:] = [d for d in dirs if not self.should_exclude(root_path / d)]
            
            for file in files:
                file_path = root_path / file
                ext = file_path.suffix.lower()
                
                if ext not in SCANNABLE_EXTENSIONS:
                    continue
                
                if self.should_exclude(file_path):
                    continue
                
                try:
                    relative = file_path.relative_to(self.project_root)
                    # Use forward slashes and remove extension for key
                    relative_str = str(relative.with_suffix('')).replace('\\', '/')
                except ValueError:
                    continue
                
                is_index = file_path.stem.lower() == 'index'
                dir_path = str(relative.parent).replace('\\', '/') if is_index else None
                
                module_info = ModuleInfo(
                    absolute_path=file_path.resolve(),
                    relative_to_root=relative_str,
                    name=file_path.stem,
                    extension=ext,
                    is_index=is_index,
                    directory=dir_path
                )
                
                # Register by full relative path
                self.modules[relative_str] = module_info
                
                # For index files, also register by directory path
                if is_index and dir_path and dir_path != '.':
                    self.modules[dir_path] = module_info
        
        self.log(f"Found {len(self.modules)} modules", "ok")
        return self.modules
    
    def scan_file(self, file_path: Path) -> List[ImportStatement]:
        """Scan a single file for all import statements."""
        imports = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding='latin-1')
            except Exception as e:
                self.log(f"Cannot read {file_path}: {e}", "error")
                return imports
        except Exception as e:
            self.log(f"Cannot read {file_path}: {e}", "error")
            return imports
        
        lines = content.split('\n')
        seen_imports = set()  # Avoid duplicates from overlapping patterns
        
        for line_num, line in enumerate(lines, 1):
            # Skip comment-only lines
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                continue
            
            for pattern in IMPORT_PATTERNS:
                for match in pattern.finditer(line):
                    import_path = match.group(2)
                    column = match.start(2)
                    
                    # Deduplicate
                    key = (line_num, column, import_path)
                    if key in seen_imports:
                        continue
                    seen_imports.add(key)
                    
                    import_type = self.resolver.classify_import(import_path)
                    
                    # Validate the import from THIS FILE's perspective
                    is_valid, resolved, issues, details = self.resolver.resolve_import(
                        import_path, file_path
                    )
                    
                    import_stmt = ImportStatement(
                        file_path=file_path,
                        line_number=line_num,
                        column=column,
                        full_line=line.rstrip(),
                        import_path=import_path,
                        import_type=import_type,
                        is_valid=is_valid,
                        resolved_path=resolved,
                        issues=issues,
                        issue_details=details
                    )
                    
                    # Generate fix suggestion for problematic imports
                    if issues or not is_valid:
                        self._suggest_fix(import_stmt)
                    
                    imports.append(import_stmt)
        
        return imports
    
    def _suggest_fix(self, import_stmt: ImportStatement):
        """Generate a fix suggestion for a problematic import."""
        original = import_stmt.import_path
        from_file = import_stmt.file_path
        
        # Priority 1: Remove .ts/.tsx extension
        if IssueType.HAS_TS_EXTENSION in import_stmt.issues:
            fixed = re.sub(r'\.(ts|tsx)$', '', original)
            # Verify the fixed path would work
            is_valid, _, _, _ = self.resolver.resolve_import(fixed, from_file)
            if is_valid:
                import_stmt.suggested_fix = fixed
                import_stmt.confidence = Confidence.HIGH
                import_stmt.fix_explanation = "Remove TypeScript extension from import"
                return
            # Even if not valid, removing extension is still correct
            import_stmt.suggested_fix = fixed
            import_stmt.confidence = Confidence.MEDIUM
            import_stmt.fix_explanation = "Remove .ts extension (file may have moved)"
            return
        
        # Priority 2: Remove explicit /index
        if IssueType.INDEX_EXPLICIT in import_stmt.issues and len(import_stmt.issues) == 1:
            fixed = re.sub(r'/index(\.(ts|tsx|js|jsx))?$', '', original)
            import_stmt.suggested_fix = fixed
            import_stmt.confidence = Confidence.HIGH
            import_stmt.fix_explanation = "Use directory import instead of explicit /index"
            return
        
        # Priority 3: Find the correct module
        # Extract target name from the import path
        clean_path = original.replace('\\', '/')
        
        # Strip ./ and ../ prefixes and any extension
        path_parts = clean_path.split('/')
        relative_parts = []
        for part in path_parts:
            if part in ('.', '..'):
                continue
            # Remove extension
            part = re.sub(r'\.(ts|tsx|js|jsx|mjs|cjs)$', '', part)
            if part:
                relative_parts.append(part)
        
        if not relative_parts:
            import_stmt.confidence = Confidence.NONE
            import_stmt.fix_explanation = "Cannot determine target module name"
            return
        
        # Try to find matching module
        target_name = relative_parts[-1]
        target_path = '/'.join(relative_parts)
        
        # Check for 'types' prefix (special handling for type directories)
        is_types_import = 'types' in relative_parts
        
        # Strategy 1: Exact path match
        if target_path in self.modules:
            module = self.modules[target_path]
            new_path = self.resolver.calculate_relative_path(module, from_file)
            if new_path:
                import_stmt.suggested_fix = new_path
                import_stmt.confidence = Confidence.HIGH
                import_stmt.fix_explanation = f"Exact match: {module.relative_to_root}"
                return
        
        # Strategy 2: Try with common prefixes (src, src_v2, lib, etc.)
        for prefix in ['src', 'src_v2', 'lib', 'app', 'packages']:
            prefixed = f"{prefix}/{target_path}"
            if prefixed in self.modules:
                module = self.modules[prefixed]
                new_path = self.resolver.calculate_relative_path(module, from_file)
                if new_path:
                    import_stmt.suggested_fix = new_path
                    import_stmt.confidence = Confidence.HIGH
                    import_stmt.fix_explanation = f"Found in {prefix}/: {module.relative_to_root}"
                    return
        
        # Strategy 3: Find by name suffix
        candidates = []
        for key, module in self.modules.items():
            # Prefer types directory for types imports
            if is_types_import:
                if key.endswith(target_path) or key == f"types/{target_name}":
                    candidates.append((key, module, 100))  # High priority
                elif 'types' in key and key.endswith(target_name):
                    candidates.append((key, module, 90))
            elif module.name == target_name:
                candidates.append((key, module, 50))
            elif key.endswith(f"/{target_name}"):
                candidates.append((key, module, 40))
        
        if candidates:
            # Sort by priority (descending), then by path length (ascending)
            candidates.sort(key=lambda x: (-x[2], len(x[0])))
            best_key, best_module, priority = candidates[0]
            
            new_path = self.resolver.calculate_relative_path(best_module, from_file)
            if new_path:
                import_stmt.suggested_fix = new_path
                
                if len(candidates) == 1 or priority >= 90:
                    import_stmt.confidence = Confidence.HIGH
                    import_stmt.fix_explanation = f"Matched: {best_key}"
                elif len(candidates) <= 3:
                    import_stmt.confidence = Confidence.MEDIUM
                    import_stmt.fix_explanation = f"Best of {len(candidates)} matches: {best_key}"
                else:
                    import_stmt.confidence = Confidence.LOW
                    import_stmt.fix_explanation = f"Guessed from {len(candidates)} candidates: {best_key}"
                return
        
        import_stmt.confidence = Confidence.NONE
        import_stmt.fix_explanation = f"No module matching '{target_name}' found in project"
    
    def scan_all(self) -> ScanResult:
        """Scan all TypeScript/JavaScript files and validate imports."""
        # First, discover all modules in the project
        self.discover_modules()
        
        # Initialize resolver with discovered modules
        self.resolver = PathResolver(
            self.project_root,
            self.tsconfig,
            self.modules
        )
        
        all_imports = []
        self._files_scanned = 0
        
        # Scan files in the scan directory
        self.log(f"Scanning imports in: {self.scan_directory}")
        
        for root, dirs, files in os.walk(self.scan_directory):
            root_path = Path(root)
            
            # Filter excluded directories
            dirs[:] = [d for d in dirs if not self.should_exclude(root_path / d)]
            
            for file in files:
                file_path = root_path / file
                
                if file_path.suffix.lower() not in SCANNABLE_EXTENSIONS:
                    continue
                
                if self.should_exclude(file_path):
                    continue
                
                self._files_scanned += 1
                file_imports = self.scan_file(file_path)
                all_imports.extend(file_imports)
        
        self.log(f"Scanned {self._files_scanned} files, found {len(all_imports)} imports", "ok")
        
        # Categorize results
        valid = []
        invalid = []
        
        for imp in all_imports:
            if imp.is_valid and not imp.issues:
                valid.append(imp)
            else:
                invalid.append(imp)
        
        # Group by issue type
        by_issue: Dict[IssueType, List[ImportStatement]] = defaultdict(list)
        for imp in invalid:
            for issue in imp.issues:
                by_issue[issue].append(imp)
        
        # Group by confidence level
        fixable_high = [i for i in invalid if i.confidence == Confidence.HIGH]
        fixable_medium = [i for i in invalid if i.confidence == Confidence.MEDIUM]
        fixable_low = [i for i in invalid if i.confidence == Confidence.LOW]
        unfixable = [i for i in invalid if i.confidence == Confidence.NONE]
        
        return ScanResult(
            project_root=self.project_root,
            scan_directory=self.scan_directory,
            tsconfig=self.tsconfig,
            modules=self.modules,
            all_imports=all_imports,
            valid_imports=valid,
            invalid_imports=invalid,
            by_issue=dict(by_issue),
            fixable_high=fixable_high,
            fixable_medium=fixable_medium,
            fixable_low=fixable_low,
            unfixable=unfixable,
            files_scanned=self._files_scanned
        )


# ============================================================================
# IMPORT FIXER
# ============================================================================

class ImportFixer:
    """Applies fixes to import statements in source files."""
    
    def __init__(
        self,
        dry_run: bool = True,
        verbose: bool = False,
        project_root: Optional[Path] = None
    ):
        self.dry_run = dry_run
        self.verbose = verbose
        self.project_root = (project_root or Path.cwd()).resolve()
        self.workdir = Path.cwd().resolve()
        
        # Track changes: file -> [(line, old, new, explanation)]
        self.changes: Dict[Path, List[Tuple[int, str, str, str]]] = defaultdict(list)
        self.applied = 0
        self.failed = 0
    
    def apply_fix(self, import_stmt: ImportStatement) -> bool:
        """Apply a single fix to an import statement."""
        if not import_stmt.suggested_fix:
            return False
        
        file_path = import_stmt.file_path
        old_import = import_stmt.import_path
        new_import = import_stmt.suggested_fix
        
        # Skip if no change needed
        if old_import == new_import:
            return True
        
        # Record the change
        self.changes[file_path].append((
            import_stmt.line_number,
            old_import,
            new_import,
            import_stmt.fix_explanation or ""
        ))
        
        if self.dry_run:
            self.applied += 1
            return True
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Replace the import path (handle both quote types)
            # Be precise to avoid replacing substrings
            for quote in ["'", '"']:
                old_pattern = f"{quote}{old_import}{quote}"
                new_pattern = f"{quote}{new_import}{quote}"
                if old_pattern in content:
                    content = content.replace(old_pattern, new_pattern)
                    break
            
            file_path.write_text(content, encoding='utf-8')
            self.applied += 1
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error fixing {file_path}: {e}")
            self.failed += 1
            return False
    
    def apply_all(
        self,
        imports: List[ImportStatement],
        confidence_levels: Set[Confidence]
    ) -> Tuple[int, int]:
        """Apply all fixes matching the specified confidence levels."""
        for imp in imports:
            if imp.confidence in confidence_levels and imp.suggested_fix:
                self.apply_fix(imp)
        return (self.applied, self.failed)
    
    def print_changes(self):
        """Print summary of all changes (applied or pending)."""
        if not self.changes:
            print("\n📭 No changes to apply.")
            return
        
        mode = "DRY-RUN - PREVIEW" if self.dry_run else "APPLIED"
        
        print("\n" + "═" * 80)
        print(f"  CHANGES ({mode})")
        print("═" * 80)
        print(f"  📂 Working directory: {self.workdir}")
        print(f"  📁 Project root:      {self.project_root}")
        print("─" * 80)
        
        for file_path, changes in sorted(self.changes.items()):
            try:
                rel_path = file_path.relative_to(self.project_root)
            except ValueError:
                rel_path = file_path
            
            print(f"\n  📄 {rel_path}")
            
            for line_num, old, new, explanation in sorted(changes, key=lambda x: x[0]):
                print(f"     L{line_num}: '{old}'")
                print(f"        → '{new}'")
                if explanation:
                    print(f"        💡 {explanation}")
        
        print("\n" + "─" * 80)
        action = "Would change" if self.dry_run else "Changed"
        print(f"  {action}: {self.applied} imports in {len(self.changes)} files")
        if self.failed > 0:
            print(f"  ❌ Failed: {self.failed}")
        print("═" * 80)


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generates human-readable and machine-readable reports."""
    
    @staticmethod
    def print_summary(result: ScanResult):
        """Print a summary of scan results."""
        print("\n" + "═" * 80)
        print("  IMPORT SCAN SUMMARY")
        print("═" * 80)
        print(f"  📁 Project root:    {result.project_root}")
        print(f"  📂 Scanned:         {result.scan_directory}")
        
        if result.tsconfig:
            try:
                tc_rel = result.tsconfig.config_path.relative_to(result.project_root)
            except ValueError:
                tc_rel = result.tsconfig.config_path
            print(f"  📋 tsconfig.json:   {tc_rel}")
            if result.tsconfig.base_url:
                try:
                    bu_rel = result.tsconfig.base_url.relative_to(result.project_root)
                except ValueError:
                    bu_rel = result.tsconfig.base_url
                print(f"     baseUrl:         {bu_rel}")
            if result.tsconfig.paths:
                print(f"     paths:           {len(result.tsconfig.paths)} aliases")
        
        print("─" * 80)
        print(f"  📦 Modules found:   {len(result.modules)}")
        print(f"  📄 Files scanned:   {result.files_scanned}")
        print(f"  📥 Total imports:   {len(result.all_imports)}")
        print()
        print(f"  ✅ Valid imports:   {len(result.valid_imports)}")
        print(f"  ❌ Invalid imports: {len(result.invalid_imports)}")
        
        if result.by_issue:
            print()
            print("  Issues by type:")
            issue_info = {
                IssueType.CANNOT_RESOLVE: ("🔍", "Cannot resolve path"),
                IssueType.MISSING_FILE: ("📄", "File not found"),
                IssueType.HAS_TS_EXTENSION: ("📎", "Has .ts extension"),
                IssueType.HAS_JS_EXTENSION: ("📎", "Has .js extension"),
                IssueType.WRONG_DEPTH: ("📏", "Wrong path depth"),
                IssueType.ALIAS_NOT_FOUND: ("🏷️", "Alias not defined"),
                IssueType.INDEX_EXPLICIT: ("📁", "Explicit /index")
            }
            for issue_type in IssueType:
                if issue_type in result.by_issue:
                    count = len(result.by_issue[issue_type])
                    icon, desc = issue_info.get(issue_type, ("❓", issue_type.value))
                    print(f"     {icon} {desc}: {count}")
        
        print()
        print("  Fixable by confidence:")
        print(f"     🟢 HIGH:   {len(result.fixable_high)}")
        print(f"     🟡 MEDIUM: {len(result.fixable_medium)}")
        print(f"     🟠 LOW:    {len(result.fixable_low)}")
        print(f"     ⚫ NONE:   {len(result.unfixable)}")
        print("═" * 80)
    
    @staticmethod
    def print_invalid_imports(result: ScanResult, show_details: bool = False):
        """Print detailed list of invalid imports."""
        if not result.invalid_imports:
            print("\n✅ All imports are valid!")
            return
        
        print("\n" + "═" * 80)
        print("  INVALID IMPORTS DETAIL")
        print("═" * 80)
        
        # Group by file
        by_file: Dict[Path, List[ImportStatement]] = defaultdict(list)
        for imp in result.invalid_imports:
            by_file[imp.file_path].append(imp)
        
        for file_path in sorted(by_file.keys()):
            imports = by_file[file_path]
            
            try:
                rel_path = file_path.relative_to(result.project_root)
            except ValueError:
                rel_path = file_path
            
            print(f"\n  📄 {rel_path}")
            
            for imp in sorted(imports, key=lambda x: x.line_number):
                conf_icon = {
                    Confidence.HIGH: "🟢",
                    Confidence.MEDIUM: "🟡",
                    Confidence.LOW: "🟠",
                    Confidence.NONE: "⚫"
                }[imp.confidence]
                
                issue_str = ", ".join(i.value for i in imp.issues)
                
                print(f"     L{imp.line_number}:{imp.column} {conf_icon} '{imp.import_path}'")
                print(f"        Issues: {issue_str}")
                
                if imp.suggested_fix:
                    print(f"        → '{imp.suggested_fix}'")
                    if imp.fix_explanation:
                        print(f"        💡 {imp.fix_explanation}")
                
                if show_details and imp.issue_details:
                    print(f"        ⚠️  {imp.issue_details}")
        
        print()
    
    @staticmethod
    def export_json(result: ScanResult, output_path: Path):
        """Export detailed results to JSON file."""
        data = {
            "version": VERSION,
            "project_root": str(result.project_root),
            "scan_directory": str(result.scan_directory),
            "tsconfig": str(result.tsconfig.config_path) if result.tsconfig else None,
            "summary": {
                "modules_found": len(result.modules),
                "files_scanned": result.files_scanned,
                "total_imports": len(result.all_imports),
                "valid_imports": len(result.valid_imports),
                "invalid_imports": len(result.invalid_imports),
                "fixable_high": len(result.fixable_high),
                "fixable_medium": len(result.fixable_medium),
                "fixable_low": len(result.fixable_low),
                "unfixable": len(result.unfixable)
            },
            "issues_by_type": {
                issue.value: len(imports)
                for issue, imports in result.by_issue.items()
            },
            "invalid_imports": [
                {
                    "file": str(imp.file_path.relative_to(result.project_root)),
                    "line": imp.line_number,
                    "column": imp.column,
                    "import_path": imp.import_path,
                    "import_type": imp.import_type.value,
                    "issues": [i.value for i in imp.issues],
                    "issue_details": imp.issue_details,
                    "suggested_fix": imp.suggested_fix,
                    "confidence": imp.confidence.value,
                    "explanation": imp.fix_explanation
                }
                for imp in result.invalid_imports
            ]
        }
        
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"\n📊 Report exported to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with comprehensive help."""
    
    parser = argparse.ArgumentParser(
        prog='adg-typescript-imports-fixer',
        description=f'''
╔══════════════════════════════════════════════════════════════════════════════════╗
║           ADG-TYPESCRIPT-IMPORTS-FIXER v{VERSION}                                 ║
║                                                                                  ║
║  Comprehensive TypeScript/JavaScript import path analyzer and fixer.            ║
║  Validates EVERY import in EVERY file from the importing file's perspective.    ║
║                                                                                  ║
║  License: {LICENSE}                                             ║
╚══════════════════════════════════════════════════════════════════════════════════╝

HOW IT WORKS:
  1. Finds tsconfig.json (or uses specified location/directory)
  2. Discovers all modules in the project
  3. Scans every file for import/export statements
  4. Validates each import FROM THE FILE'S LOCATION
  5. Suggests fixes with confidence levels

DEFAULT BEHAVIOR:
  Without arguments, looks for tsconfig.json starting from current directory,
  uses its location as project root, and scans for imports.

EXAMPLES:
  # Auto-detect tsconfig.json and scan
  python adg-typescript-imports-fixer2.py

  # Use specific tsconfig.json
  python adg-typescript-imports-fixer2.py --tsconfig ./tsconfig.json

  # Scan specific directory (finds tsconfig automatically)
  python adg-typescript-imports-fixer2.py --dir ./src

  # Scan with explicit project root and scan directory
  python adg-typescript-imports-fixer2.py --tsconfig ./tsconfig.json --dir ./src_v2

  # Preview HIGH confidence fixes (safe)
  python adg-typescript-imports-fixer2.py --fix-high --dry-run

  # Apply HIGH confidence fixes
  python adg-typescript-imports-fixer2.py --fix-high

  # Apply all fixable imports (review carefully!)
  python adg-typescript-imports-fixer2.py --fix-all

  # Export detailed report to JSON
  python adg-typescript-imports-fixer2.py --export report.json

  # List all discovered modules
  python adg-typescript-imports-fixer2.py --list-modules
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
DETECTED ISSUES:
  🔍 cannot_resolve   - Import path doesn't resolve to any file
  📄 missing_file     - Directory exists but target file was deleted/moved
  📎 has_ts_extension - Import has .ts/.tsx extension (anti-pattern in TypeScript)
  📏 wrong_depth      - Path escapes project root (too many ../)
  🏷️  alias_not_found  - @alias not defined in tsconfig.json paths
  📁 index_explicit   - Uses /index instead of directory import

CONFIDENCE LEVELS:
  🟢 HIGH   - Exact module match or extension removal. Safe to auto-fix.
  🟡 MEDIUM - Single partial match. Review recommended.
  🟠 LOW    - Multiple matches, best guess. Manual review needed.
  ⚫ NONE   - No suggestion available. Manual fix required.

SAFETY TIPS:
  • Always use --dry-run first to preview changes
  • Backup your code before using --fix-* without --dry-run
  • Start with --fix-high (safest fixes)
  • Review --fix-medium and --fix-low carefully

{LICENSE} - https://mozilla.org/MPL/2.0/
'''
    )
    
    # Target selection
    target = parser.add_argument_group('Target Selection')
    target.add_argument(
        '--tsconfig', '-c',
        type=Path,
        metavar='FILE',
        help='Path to tsconfig.json. If not specified, searches from --dir or current directory upward.'
    )
    target.add_argument(
        '--dir', '-d',
        type=Path,
        metavar='DIRECTORY',
        help='Directory to scan for imports. Default: tsconfig.json location or current directory.'
    )
    target.add_argument(
        '--exclude', '-e',
        action='append',
        metavar='PATTERN',
        help='Additional patterns to exclude. Can be used multiple times. Example: -e "*.test.ts" -e "BURDEL"'
    )
    target.add_argument(
        '--no-exclude',
        action='store_true',
        help='Disable default excludes (node_modules, dist, etc.). Use with caution!'
    )
    
    # Fix options
    fix = parser.add_argument_group('Fix Options')
    fix.add_argument(
        '--fix-high',
        action='store_true',
        help='Apply HIGH confidence fixes. Safest option - exact matches and extension removal.'
    )
    fix.add_argument(
        '--fix-medium',
        action='store_true',
        help='Apply MEDIUM confidence fixes. Single partial matches - review recommended.'
    )
    fix.add_argument(
        '--fix-low',
        action='store_true',
        help='Apply LOW confidence fixes. Multiple matches - may guess wrong!'
    )
    fix.add_argument(
        '--fix-all',
        action='store_true',
        help='Apply ALL fixes (HIGH + MEDIUM + LOW). Review with --dry-run first!'
    )
    fix.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Preview changes without modifying files. Always use this first!'
    )
    
    # Output options
    output = parser.add_argument_group('Output Options')
    output.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output including issue details and scanning progress.'
    )
    output.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output - only errors and final summary.'
    )
    output.add_argument(
        '--export',
        type=Path,
        metavar='FILE',
        help='Export detailed results to JSON file for CI/CD or further processing.'
    )
    output.add_argument(
        '--list-modules',
        action='store_true',
        help='List all discovered modules in the project.'
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
    args = parser.parse_args()
    
    # If no arguments, still try to run with auto-detection
    tsconfig: Optional[TsConfig] = None
    tsconfig_path: Optional[Path] = None
    
    # Step 1: Find or load tsconfig.json
    if args.tsconfig:
        tsconfig_path = args.tsconfig.resolve()
        if not tsconfig_path.exists():
            print(f"❌ Error: tsconfig.json not found: {tsconfig_path}")
            sys.exit(1)
    else:
        # Auto-detect from --dir or current directory
        search_from = args.dir.resolve() if args.dir else Path.cwd()
        tsconfig_path = TsConfigParser.find_tsconfig(search_from)
    
    if tsconfig_path:
        try:
            tsconfig = TsConfigParser.parse(tsconfig_path)
            if not args.quiet:
                print(f"📋 Using tsconfig: {tsconfig_path}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to parse tsconfig.json: {e}")
            print("   Continuing without tsconfig configuration...")
    else:
        if not args.quiet:
            print("ℹ️  No tsconfig.json found. Using defaults.")
    
    # Step 2: Determine project root and scan directory
    if tsconfig:
        project_root = tsconfig.project_root
    elif args.dir:
        project_root = args.dir.resolve()
    else:
        project_root = Path.cwd().resolve()
    
    scan_directory = args.dir.resolve() if args.dir else project_root
    
    # Validate paths
    if not project_root.exists():
        print(f"❌ Error: Project root does not exist: {project_root}")
        sys.exit(1)
    
    if not scan_directory.exists():
        print(f"❌ Error: Scan directory does not exist: {scan_directory}")
        sys.exit(1)
    
    # Step 3: Create scanner and scan
    scanner = ImportScanner(
        project_root=project_root,
        scan_directory=scan_directory,
        tsconfig=tsconfig,
        verbose=args.verbose,
        exclude_patterns=args.exclude,
        no_exclude=args.no_exclude
    )
    
    if not args.quiet:
        if project_root != scan_directory:
            print(f"📁 Project root: {project_root}")
            print(f"🔍 Scanning: {scan_directory}")
        else:
            print(f"🔍 Scanning: {project_root}")
    
    result = scanner.scan_all()
    
    # Step 4: Output results
    if args.list_modules:
        print("\n" + "═" * 60)
        print("  DISCOVERED MODULES")
        print("═" * 60)
        for key in sorted(result.modules.keys()):
            module = result.modules[key]
            idx = " [index]" if module.is_index else ""
            print(f"  {key}{idx}")
        print()
    
    if not args.quiet:
        ReportGenerator.print_summary(result)
        ReportGenerator.print_invalid_imports(result, show_details=args.verbose)
    
    if args.export:
        ReportGenerator.export_json(result, args.export)
    
    # Step 5: Apply fixes if requested
    should_fix = args.fix_high or args.fix_medium or args.fix_low or args.fix_all
    
    if should_fix:
        levels: Set[Confidence] = set()
        if args.fix_high or args.fix_all:
            levels.add(Confidence.HIGH)
        if args.fix_medium or args.fix_all:
            levels.add(Confidence.MEDIUM)
        if args.fix_low or args.fix_all:
            levels.add(Confidence.LOW)
        
        # Confirmation if not dry-run
        if not args.dry_run and levels:
            level_str = ", ".join(sorted(l.value.upper() for l in levels))
            print(f"\n⚠️  About to apply {level_str} confidence fixes.")
            print("   This will modify your source files!")
            print("   Make sure you have a backup or use version control.")
            try:
                response = input("   Continue? [y/N]: ")
                if response.lower() != 'y':
                    print("❌ Aborted.")
                    sys.exit(0)
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Aborted.")
                sys.exit(0)
        
        # Create fixer and apply
        fixer = ImportFixer(
            dry_run=args.dry_run,
            verbose=args.verbose,
            project_root=project_root
        )
        
        imports_to_fix = [
            i for i in result.invalid_imports
            if i.confidence in levels and i.suggested_fix
        ]
        
        if imports_to_fix:
            for imp in imports_to_fix:
                fixer.apply_fix(imp)
            fixer.print_changes()
        else:
            print("\n📭 No fixes to apply for selected confidence levels.")
    
    # Exit code
    if result.unfixable:
        sys.exit(1)  # There are unfixable issues
    elif result.invalid_imports:
        sys.exit(2)  # There are issues but all fixable
    else:
        sys.exit(0)  # All good!


if __name__ == '__main__':
    # Ensure UTF-8 output on Windows
    if sys.platform == 'win32':
        import io
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except AttributeError:
            pass  # Already wrapped or not a real terminal
    
    main()
