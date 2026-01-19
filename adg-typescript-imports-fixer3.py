#!/usr/bin/env python3
"""
fix-imports.py v2.0 - TypeScript/JavaScript Import Path Analyzer & Fixer

A comprehensive tool for analyzing and fixing import paths in TypeScript/JavaScript 
projects. Works relative to tsconfig.json, validates every import in every file,
and provides detailed diagnostics.

Features:
- Reads and respects tsconfig.json (paths, baseUrl, include, exclude)
- Validates EVERY import statement in EVERY file
- Detects missing files (path exists but file doesn't)
- Detects leftover .ts extensions in imports
- Tests paths from the importing file's perspective
- Multiple confidence levels for auto-fixes
- Dry-run mode for safe previewing

Author: ADG-Parallels2 Team
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

# File extensions to scan
SCANNABLE_EXTENSIONS = {'.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs'}

# Extensions to try when resolving imports
RESOLVE_EXTENSIONS = ['.ts', '.tsx', '.js', '.jsx', '/index.ts', '/index.tsx', '/index.js']

# Import/export patterns - capture full statement for better context
# Note: Use re.DOTALL for patterns that need to match across newlines
IMPORT_PATTERNS = [
    # import { x, y, z } from 'path' - multiline support with DOTALL
    re.compile(r'''(import\s+\{[^}]*\}\s+from\s+['"])([^'"]+)(['"])''', re.DOTALL),
    # import x from 'path'
    re.compile(r'''(import\s+[\w]+\s+from\s+['"])([^'"]+)(['"])'''),
    # import * as x from 'path'
    re.compile(r'''(import\s+\*\s+as\s+\w+\s+from\s+['"])([^'"]+)(['"])'''),
    # import 'path' (side-effect)
    re.compile(r'''(import\s+['"])([^'"]+)(['"])'''),
    # export { x } from 'path' - multiline support
    re.compile(r'''(export\s+\{[^}]*\}\s+from\s+['"])([^'"]+)(['"])''', re.DOTALL),
    # export * from 'path'
    re.compile(r'''(export\s+\*\s+from\s+['"])([^'"]+)(['"])'''),
    # require('path')
    re.compile(r'''(require\s*\(\s*['"])([^'"]+)(['"]\s*\))'''),
    # import type { x } from 'path' - multiline support
    re.compile(r'''(import\s+type\s+\{[^}]*\}\s+from\s+['"])([^'"]+)(['"])''', re.DOTALL),
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


class IssueType(Enum):
    """Type of import issue detected."""
    CANNOT_RESOLVE = "cannot_resolve"       # Path doesn't resolve to any file
    MISSING_FILE = "missing_file"           # Path structure exists but file doesn't
    HAS_EXTENSION = "has_extension"         # Import has .ts/.tsx extension (bad practice)
    WRONG_DEPTH = "wrong_depth"             # Too many or too few ../
    ALIAS_NOT_FOUND = "alias_not_found"     # @alias not in tsconfig paths


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


@dataclass
class ModuleInfo:
    """Information about a discovered module/file."""
    absolute_path: Path           # Absolute path to file
    relative_to_root: str         # Relative to project root (forward slashes)
    name: str                     # Module name (without extension)
    extension: str                # File extension
    is_index: bool = False        # True if this is an index.ts file
    directory: Optional[str] = None  # Parent directory for index files


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


# ============================================================================
# TSCONFIG PARSER
# ============================================================================

class TsConfigParser:
    """Parses tsconfig.json and resolves extends."""
    
    @staticmethod
    def find_tsconfig(start_path: Path) -> Optional[Path]:
        """Find tsconfig.json by walking up the directory tree."""
        current = start_path.resolve()
        while current != current.parent:
            tsconfig_path = current / 'tsconfig.json'
            if tsconfig_path.exists():
                return tsconfig_path
            current = current.parent
        return None
    
    @staticmethod
    def parse(config_path: Path) -> TsConfig:
        """Parse tsconfig.json file."""
        if not config_path.exists():
            raise FileNotFoundError(f"tsconfig.json not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # JSONC parsing: remove comments and trailing commas
            # Must be careful not to touch content inside strings
            
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
                    i += 2  # Skip the closing */
                else:
                    result.append(char)
                    i += 1
            
            content = ''.join(result)
            
            # Remove trailing commas (JSONC feature)
            content = re.sub(r',(\s*[}\]])', r'\1', content)
            
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in tsconfig.json: {e}")
        
        config_dir = config_path.parent.resolve()
        compiler_options = data.get('compilerOptions', {})
        
        # Parse baseUrl
        base_url = None
        if 'baseUrl' in compiler_options:
            base_url = (config_dir / compiler_options['baseUrl']).resolve()
        
        # Parse paths
        paths = {}
        for alias, targets in compiler_options.get('paths', {}).items():
            # Normalize alias (remove trailing /*)
            paths[alias] = targets
        
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
    """Resolves import paths to actual files."""
    
    def __init__(
        self, 
        project_root: Path,
        tsconfig: Optional[TsConfig] = None,
        modules: Optional[Dict[str, ModuleInfo]] = None
    ):
        self.project_root = project_root.resolve()
        self.tsconfig = tsconfig
        self.modules = modules or {}
        self.base_url = tsconfig.base_url if tsconfig else project_root
    
    def classify_import(self, import_path: str) -> ImportType:
        """Classify the type of import path."""
        if import_path.startswith('./') or import_path.startswith('../'):
            return ImportType.RELATIVE
        elif import_path.startswith('/'):
            return ImportType.ABSOLUTE
        elif import_path.startswith('@'):
            # Check if it's a tsconfig alias or a scoped package
            if self.tsconfig and self.tsconfig.paths:
                for alias in self.tsconfig.paths.keys():
                    alias_base = alias.replace('/*', '')
                    if import_path.startswith(alias_base):
                        return ImportType.ALIAS
            # Check common scoped packages
            scoped_packages = ['@types/', '@babel/', '@jest/', '@testing-library/', 
                              '@angular/', '@vue/', '@react/', '@nestjs/']
            for pkg in scoped_packages:
                if import_path.startswith(pkg):
                    return ImportType.PACKAGE
            return ImportType.ALIAS  # Assume alias if starts with @
        else:
            return ImportType.PACKAGE
    
    def resolve_alias(self, import_path: str) -> Optional[str]:
        """Resolve a path alias to actual path."""
        if not self.tsconfig or not self.tsconfig.paths:
            return None
        
        for alias, targets in self.tsconfig.paths.items():
            alias_base = alias.replace('/*', '')
            if import_path.startswith(alias_base):
                # Get the rest of the path after alias
                rest = import_path[len(alias_base):].lstrip('/')
                
                # Try each target
                for target in targets:
                    target_base = target.replace('/*', '')
                    resolved = target_base + ('/' + rest if rest else '')
                    return resolved
        
        return None
    
    def resolve_import(
        self, 
        import_path: str, 
        from_file: Path
    ) -> Tuple[bool, Optional[Path], List[IssueType], Optional[str]]:
        """
        Resolve an import path to an actual file.
        
        Returns: (is_valid, resolved_path, issues, details)
        """
        issues = []
        details = None
        
        import_type = self.classify_import(import_path)
        
        # Check for .ts extension in import (bad practice)
        if import_path.endswith('.ts') or import_path.endswith('.tsx'):
            issues.append(IssueType.HAS_EXTENSION)
        
        # Skip package imports - can't validate without node_modules
        if import_type == ImportType.PACKAGE:
            return (True, None, issues, "Package import - skipped")
        
        # Handle alias imports
        if import_type == ImportType.ALIAS:
            resolved_alias = self.resolve_alias(import_path)
            if resolved_alias is None:
                issues.append(IssueType.ALIAS_NOT_FOUND)
                return (False, None, issues, f"Alias not found in tsconfig.paths")
            # Convert alias to relative path from base_url
            target = self.base_url / resolved_alias
        elif import_type == ImportType.RELATIVE:
            # Resolve relative to importing file's directory
            from_dir = from_file.parent
            target = (from_dir / import_path).resolve()
        else:
            # Absolute path
            target = Path(import_path)
        
        # Try to find the actual file
        resolved = self._find_file(target)
        
        if resolved:
            return (True, resolved, issues, None)
        
        # File not found - determine why
        issues.append(IssueType.CANNOT_RESOLVE)
        
        # Check if directory exists but file doesn't
        target_dir = target.parent
        if target_dir.exists():
            issues.append(IssueType.MISSING_FILE)
            details = f"Directory exists but file not found: {target.name}"
        else:
            # Check if too many ../ (went above project root)
            try:
                target.relative_to(self.project_root)
            except ValueError:
                issues.append(IssueType.WRONG_DEPTH)
                details = f"Path escapes project root"
        
        if not details:
            details = f"Cannot resolve to: {target}"
        
        return (False, None, issues, details)
    
    def _find_file(self, target: Path) -> Optional[Path]:
        """Try to find actual file with various extensions."""
        # Direct match (if already has extension)
        if target.exists() and target.is_file():
            return target
        
        # Try extensions
        for ext in RESOLVE_EXTENSIONS:
            candidate = Path(str(target) + ext)
            if candidate.exists() and candidate.is_file():
                return candidate
        
        # Try as directory with index
        if target.exists() and target.is_dir():
            for index_ext in ['.ts', '.tsx', '.js', '.jsx']:
                index_file = target / f'index{index_ext}'
                if index_file.exists():
                    return index_file
        
        return None
    
    def calculate_correct_path(
        self,
        target_module: ModuleInfo,
        from_file: Path
    ) -> Optional[str]:
        """Calculate the correct relative path from from_file to target_module."""
        try:
            from_dir = from_file.parent
            target = target_module.absolute_path
            
            # If it's an index file, use parent directory
            if target_module.is_index:
                target = target.parent
            
            rel_path = os.path.relpath(target, from_dir)
            rel_path = rel_path.replace('\\', '/')
            
            # Add ./ if needed
            if not rel_path.startswith('.'):
                rel_path = './' + rel_path
            
            # Remove .ts extension
            rel_path = re.sub(r'\.tsx?$', '', rel_path)
            
            # Remove /index suffix
            rel_path = re.sub(r'/index$', '', rel_path)
            
            return rel_path
        except ValueError:
            return None


# ============================================================================
# IMPORT SCANNER
# ============================================================================

class ImportScanner:
    """Scans TypeScript/JavaScript files for import statements."""
    
    def __init__(
        self,
        project_root: Path,
        tsconfig: Optional[TsConfig] = None,
        verbose: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        no_exclude: bool = False,
        scan_dir: Optional[Path] = None  # Directory to scan for imports (default: project_root)
    ):
        self.project_root = project_root.resolve()
        self.tsconfig = tsconfig
        self.verbose = verbose
        # scan_dir is where we look for import statements to validate
        # project_root is where we discover all available modules
        self.scan_dir = (scan_dir or project_root).resolve()
        
        # Build exclude list from tsconfig and custom patterns
        if no_exclude:
            # Only use explicitly provided patterns
            self.exclude_patterns = set(exclude_patterns or [])
        else:
            self.exclude_patterns = set(exclude_patterns or [])
            if tsconfig:
                self.exclude_patterns.update(tsconfig.exclude)
            self.exclude_patterns.update(['node_modules', '.git', 'dist', 'build', 'out'])
        
        self.modules: Dict[str, ModuleInfo] = {}
        self.resolver: Optional[PathResolver] = None
    
    def log(self, message: str, level: str = "info"):
        """Log message if verbose mode is on."""
        if self.verbose:
            prefix = {"info": "ℹ️ ", "warn": "⚠️ ", "error": "❌", "ok": "✅"}
            print(f"{prefix.get(level, '')} {message}")
    
    def should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded from scanning."""
        path_str = str(path)
        for pattern in self.exclude_patterns:
            if pattern in path_str or pattern in path.parts:
                return True
        return False
    
    def discover_modules(self) -> Dict[str, ModuleInfo]:
        """Discover all TypeScript/JavaScript modules in the project."""
        self.log(f"Discovering modules in: {self.project_root}")
        
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            
            # Filter excluded directories
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
                    relative_str = str(relative.with_suffix('')).replace('\\', '/')
                except ValueError:
                    continue
                
                is_index = file_path.stem == 'index'
                
                module_info = ModuleInfo(
                    absolute_path=file_path,
                    relative_to_root=relative_str,
                    name=file_path.stem,
                    extension=ext,
                    is_index=is_index,
                    directory=str(relative.parent).replace('\\', '/') if is_index else None
                )
                
                # Register by relative path
                self.modules[relative_str] = module_info
                
                # For index files, also register by directory
                if is_index and module_info.directory and module_info.directory != '.':
                    self.modules[module_info.directory] = module_info
        
        self.log(f"Found {len(self.modules)} modules")
        return self.modules
    
    def scan_file(self, file_path: Path) -> List[ImportStatement]:
        """Scan a single file for all import statements."""
        imports = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            self.log(f"Cannot read {file_path}: {e}", "error")
            return imports
        
        # Build line number lookup for positions in content
        lines = content.split('\n')
        line_starts = [0]
        for line in lines[:-1]:
            line_starts.append(line_starts[-1] + len(line) + 1)  # +1 for newline
        
        def pos_to_line_col(pos: int) -> Tuple[int, int]:
            """Convert character position to (line_number, column)."""
            for i, start in enumerate(line_starts):
                if i + 1 < len(line_starts):
                    if start <= pos < line_starts[i + 1]:
                        return (i + 1, pos - start)
                else:
                    return (i + 1, pos - start)
            return (len(lines), 0)
        
        # Scan entire content (handles multiline imports)
        seen_imports = set()  # Avoid duplicates from overlapping patterns
        
        for pattern in IMPORT_PATTERNS:
            for match in pattern.finditer(content):
                import_path = match.group(2)
                match_start = match.start(2)
                
                # Deduplicate based on position and path
                key = (match_start, import_path)
                if key in seen_imports:
                    continue
                seen_imports.add(key)
                
                line_num, column = pos_to_line_col(match_start)
                full_line = lines[line_num - 1] if line_num <= len(lines) else ""
                
                import_type = self.resolver.classify_import(import_path)
                
                # Validate the import
                is_valid, resolved, issues, details = self.resolver.resolve_import(
                    import_path, file_path
                )
                
                import_stmt = ImportStatement(
                    file_path=file_path,
                    line_number=line_num,
                    column=column,
                    full_line=full_line.strip(),
                    import_path=import_path,
                    import_type=import_type,
                    is_valid=is_valid and len(issues) == 0,
                    resolved_path=resolved,
                    issues=issues,
                    issue_details=details
                )
                
                # Generate fix suggestion for invalid imports
                if not import_stmt.is_valid or issues:
                    self._suggest_fix(import_stmt)
                
                imports.append(import_stmt)
        
        return imports
    
    def _suggest_fix(self, import_stmt: ImportStatement):
        """Generate fix suggestion for an invalid import."""
        original = import_stmt.import_path
        from_file = import_stmt.file_path
        
        # Handle .ts extension issue
        if IssueType.HAS_EXTENSION in import_stmt.issues:
            # Simply remove the extension
            fixed = re.sub(r'\.tsx?$', '', original)
            import_stmt.suggested_fix = fixed
            import_stmt.confidence = Confidence.HIGH
            import_stmt.fix_explanation = "Remove .ts extension from import"
            return
        
        # Extract the target module name from the path
        # e.g., '../../types/mcp' -> 'mcp', '../../types' -> 'types' or 'index'
        clean_path = original.replace('\\', '/')
        path_parts = clean_path.split('/')
        
        # Remove leading relative markers
        while path_parts and path_parts[0] in ('.', '..'):
            path_parts = path_parts[1:]
        
        if not path_parts:
            import_stmt.confidence = Confidence.NONE
            import_stmt.fix_explanation = "Cannot determine target module"
            return
        
        # Check if it's a types import (contains 'types' in path)
        is_types_import = 'types' in path_parts
        
        # Get the actual target name
        target_name = path_parts[-1] if path_parts else ''
        
        # For paths like '../../types' -> look for types/index or just 'types'
        if target_name == 'types':
            # Looking for types/index
            search_patterns = ['types/index', 'types', 'src_v2/types/index', 'src_v2/types']
        elif is_types_import:
            # Looking for types/xxx
            search_patterns = [
                f'types/{target_name}',
                f'src_v2/types/{target_name}',
                target_name
            ]
        else:
            search_patterns = ['/'.join(path_parts), target_name]
        
        # Try to find matching module
        found_module = None
        for pattern in search_patterns:
            if pattern in self.modules:
                found_module = self.modules[pattern]
                break
        
        # If not found by exact match, try ending match
        if not found_module:
            for key, module in self.modules.items():
                # For types imports, prefer types/ directory
                if is_types_import:
                    if key.endswith(f'types/{target_name}') or key == f'types/{target_name}':
                        found_module = module
                        break
                else:
                    if key.endswith(f'/{target_name}') or key == target_name:
                        found_module = module
                        break
        
        if found_module:
            new_path = self.resolver.calculate_correct_path(found_module, from_file)
            if new_path:
                import_stmt.suggested_fix = new_path
                import_stmt.confidence = Confidence.HIGH
                import_stmt.fix_explanation = f"Found module: {found_module.relative_to_root}"
                return
        
        # Fallback: try finding by just the last component in all modules
        candidates = []
        for key, module in self.modules.items():
            if module.name == target_name or key.endswith(f'/{target_name}'):
                candidates.append((key, module))
        
        if len(candidates) == 1:
            module = candidates[0][1]
            new_path = self.resolver.calculate_correct_path(module, from_file)
            if new_path:
                import_stmt.suggested_fix = new_path
                import_stmt.confidence = Confidence.MEDIUM
                import_stmt.fix_explanation = f"Single match: {candidates[0][0]}"
                return
        
        elif len(candidates) > 1:
            # Prefer types/ match for types imports
            if is_types_import:
                types_candidates = [(k, m) for k, m in candidates if 'types' in k]
                if len(types_candidates) == 1:
                    module = types_candidates[0][1]
                    new_path = self.resolver.calculate_correct_path(module, from_file)
                    if new_path:
                        import_stmt.suggested_fix = new_path
                        import_stmt.confidence = Confidence.HIGH
                        import_stmt.fix_explanation = f"Types match: {types_candidates[0][0]}"
                        return
            
            # Pick shortest matching path
            candidates.sort(key=lambda x: len(x[0]))
            module = candidates[0][1]
            new_path = self.resolver.calculate_correct_path(module, from_file)
            if new_path:
                import_stmt.suggested_fix = new_path
                import_stmt.confidence = Confidence.LOW
                import_stmt.fix_explanation = f"Multiple matches ({len(candidates)}), picked: {candidates[0][0]}"
                return
        
        import_stmt.confidence = Confidence.NONE
        import_stmt.fix_explanation = f"No module matching '{target_name}' found"
    
    def scan_all(self) -> ScanResult:
        """Scan all files in the scan directory."""
        # Discover modules in project root first (includes all available modules)
        self.discover_modules()
        
        # Create resolver
        self.resolver = PathResolver(
            self.project_root,
            self.tsconfig,
            self.modules
        )
        
        all_imports = []
        
        # Scan files in scan_dir (which may be different from project_root)
        for root, dirs, files in os.walk(self.scan_dir):
            root_path = Path(root)
            # For scan_dir, we still respect exclude patterns
            dirs[:] = [d for d in dirs if not self.should_exclude(root_path / d)]
            
            for file in files:
                file_path = root_path / file
                if file_path.suffix.lower() in SCANNABLE_EXTENSIONS:
                    if not self.should_exclude(file_path):
                        file_imports = self.scan_file(file_path)
                        all_imports.extend(file_imports)
        
        # Categorize results
        valid = [i for i in all_imports if i.is_valid and not i.issues]
        invalid = [i for i in all_imports if not i.is_valid or i.issues]
        
        # Group by issue type
        by_issue: Dict[IssueType, List[ImportStatement]] = defaultdict(list)
        for imp in invalid:
            for issue in imp.issues:
                by_issue[issue].append(imp)
        
        # Group by confidence
        fixable_high = [i for i in invalid if i.confidence == Confidence.HIGH]
        fixable_medium = [i for i in invalid if i.confidence == Confidence.MEDIUM]
        fixable_low = [i for i in invalid if i.confidence == Confidence.LOW]
        unfixable = [i for i in invalid if i.confidence == Confidence.NONE]
        
        return ScanResult(
            project_root=self.project_root,
            tsconfig=self.tsconfig,
            modules=self.modules,
            all_imports=all_imports,
            valid_imports=valid,
            invalid_imports=invalid,
            by_issue=dict(by_issue),
            fixable_high=fixable_high,
            fixable_medium=fixable_medium,
            fixable_low=fixable_low,
            unfixable=unfixable
        )


# ============================================================================
# IMPORT FIXER
# ============================================================================

class ImportFixer:
    """Applies fixes to import statements."""
    
    def __init__(
        self,
        dry_run: bool = True,
        verbose: bool = False,
        project_root: Optional[Path] = None
    ):
        self.dry_run = dry_run
        self.verbose = verbose
        self.project_root = project_root or Path.cwd()
        self.workdir = Path.cwd()
        self.changes: Dict[Path, List[Tuple[int, str, str, str]]] = defaultdict(list)
        self.applied = 0
        self.failed = 0
    
    def apply_fix(self, import_stmt: ImportStatement) -> bool:
        """Apply a single fix."""
        if not import_stmt.suggested_fix:
            return False
        
        file_path = import_stmt.file_path
        old_import = import_stmt.import_path
        new_import = import_stmt.suggested_fix
        
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
            
            # Replace the import (handle both quote types)
            for quote in ["'", '"']:
                old_pattern = f"{quote}{old_import}{quote}"
                new_pattern = f"{quote}{new_import}{quote}"
                content = content.replace(old_pattern, new_pattern)
            
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
        """Apply all fixes matching confidence levels."""
        for imp in imports:
            if imp.confidence in confidence_levels and imp.suggested_fix:
                self.apply_fix(imp)
        return (self.applied, self.failed)
    
    def print_changes(self):
        """Print summary of all changes."""
        if not self.changes:
            print("\nNo changes to apply.")
            return
        
        mode = "DRY-RUN" if self.dry_run else "APPLIED"
        
        print("\n" + "=" * 80)
        print(f"CHANGES ({mode})")
        print("=" * 80)
        print(f"📂 Working dir:  {self.workdir}")
        print(f"📁 Project root: {self.project_root}")
        print("-" * 80)
        
        for file_path, changes in sorted(self.changes.items()):
            try:
                rel_to_project = file_path.relative_to(self.project_root)
            except ValueError:
                rel_to_project = file_path
            
            try:
                rel_to_workdir = file_path.relative_to(self.workdir)
            except ValueError:
                rel_to_workdir = file_path
            
            print(f"\n📄 {rel_to_project}")
            if str(rel_to_project) != str(rel_to_workdir):
                print(f"   (from workdir: {rel_to_workdir})")
            
            for line_num, old, new, explanation in changes:
                print(f"   L{line_num}: '{old}'")
                print(f"       → '{new}'")
                if explanation:
                    print(f"       💡 {explanation}")
        
        print("\n" + "-" * 80)
        action = "Would change" if self.dry_run else "Changed"
        print(f"{action}: {self.applied} imports in {len(self.changes)} files")
        if self.failed > 0:
            print(f"Failed: {self.failed}")


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generates reports from scan results."""
    
    @staticmethod
    def print_summary(result: ScanResult):
        """Print a summary of scan results."""
        print("\n" + "=" * 80)
        print("IMPORT SCAN SUMMARY")
        print("=" * 80)
        print(f"📁 Project root: {result.project_root}")
        if result.tsconfig:
            print(f"📋 tsconfig:     {result.tsconfig.config_path}")
            if result.tsconfig.base_url:
                print(f"   baseUrl:      {result.tsconfig.base_url}")
            if result.tsconfig.paths:
                print(f"   paths:        {len(result.tsconfig.paths)} aliases defined")
        print(f"📦 Modules found: {len(result.modules)}")
        print(f"📥 Total imports: {len(result.all_imports)}")
        print()
        print(f"✅ Valid imports:   {len(result.valid_imports)}")
        print(f"❌ Invalid imports: {len(result.invalid_imports)}")
        
        if result.by_issue:
            print()
            print("Issues by type:")
            issue_icons = {
                IssueType.CANNOT_RESOLVE: "🔍",
                IssueType.MISSING_FILE: "📄",
                IssueType.HAS_EXTENSION: "📎",
                IssueType.WRONG_DEPTH: "📏",
                IssueType.ALIAS_NOT_FOUND: "🏷️"
            }
            for issue_type, imports in sorted(result.by_issue.items(), key=lambda x: -len(x[1])):
                icon = issue_icons.get(issue_type, "❓")
                print(f"  {icon} {issue_type.value}: {len(imports)}")
        
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
        
        print("\n" + "=" * 80)
        print("INVALID IMPORTS DETAIL")
        print("=" * 80)
        
        # Group by file
        by_file: Dict[Path, List[ImportStatement]] = defaultdict(list)
        for imp in result.invalid_imports:
            by_file[imp.file_path].append(imp)
        
        for file_path, imports in sorted(by_file.items()):
            try:
                rel_path = file_path.relative_to(result.project_root)
            except ValueError:
                rel_path = file_path
            
            print(f"\n📄 {rel_path}")
            
            # Sort by line number
            for imp in sorted(imports, key=lambda x: x.line_number):
                conf_icon = {
                    Confidence.HIGH: "🟢",
                    Confidence.MEDIUM: "🟡", 
                    Confidence.LOW: "🟠",
                    Confidence.NONE: "⚫"
                }[imp.confidence]
                
                issue_str = ", ".join(i.value for i in imp.issues)
                
                print(f"   L{imp.line_number}:{imp.column} {conf_icon} '{imp.import_path}'")
                print(f"      Issues: {issue_str}")
                
                if imp.suggested_fix:
                    print(f"      → '{imp.suggested_fix}'")
                    if imp.fix_explanation:
                        print(f"      💡 {imp.fix_explanation}")
                
                if imp.issue_details and show_all:
                    print(f"      ⚠️  {imp.issue_details}")
    
    @staticmethod
    def export_json(result: ScanResult, output_path: Path):
        """Export results to JSON file."""
        data = {
            "project_root": str(result.project_root),
            "tsconfig": str(result.tsconfig.config_path) if result.tsconfig else None,
            "summary": {
                "total_modules": len(result.modules),
                "total_imports": len(result.all_imports),
                "valid_imports": len(result.valid_imports),
                "invalid_imports": len(result.invalid_imports),
                "fixable_high": len(result.fixable_high),
                "fixable_medium": len(result.fixable_medium),
                "fixable_low": len(result.fixable_low),
                "unfixable": len(result.unfixable)
            },
            "issues_by_type": {
                issue_type.value: len(imports)
                for issue_type, imports in result.by_issue.items()
            },
            "invalid_imports": [
                {
                    "file": str(imp.file_path.relative_to(result.project_root)),
                    "line": imp.line_number,
                    "column": imp.column,
                    "import": imp.import_path,
                    "issues": [i.value for i in imp.issues],
                    "suggested": imp.suggested_fix,
                    "confidence": imp.confidence.value,
                    "explanation": imp.fix_explanation
                }
                for imp in result.invalid_imports
            ]
        }
        
        output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        print(f"\n📊 Report exported to: {output_path}")


# ============================================================================
# TSC COMPARISON & DEBUG HELPERS  
# ============================================================================

def run_tsc(directory: Path) -> List[Tuple[str, int, str]]:
    """Run tsc --noEmit and parse TS2307 errors.
    
    Returns: List of (file, line, module) tuples for cannot-find-module errors.
    """
    import subprocess
    
    try:
        result = subprocess.run(
            ['npx', 'tsc', '--noEmit'],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=120
        )
        output = result.stdout + result.stderr
    except Exception as e:
        print(f"❌ Failed to run tsc: {e}")
        return []
    
    # Parse TS2307 errors: "file.ts(line,col): error TS2307: Cannot find module 'xxx'"
    pattern = re.compile(r"^(.+?)\((\d+),\d+\): error TS2307: Cannot find module ['\"]([^'\"]+)['\"]")
    
    errors = []
    for line in output.split('\n'):
        match = pattern.match(line.strip())
        if match:
            file_path = match.group(1).replace('\\', '/')
            line_num = int(match.group(2))
            module = match.group(3)
            errors.append((file_path, line_num, module))
    
    return errors


def compare_with_tsc(directory: Path, result: ScanResult):
    """Compare our scan results with tsc --noEmit output."""
    print("\n" + "=" * 80)
    print("TSC COMPARISON")
    print("=" * 80)
    print("Running tsc --noEmit...")
    
    tsc_errors = run_tsc(directory)
    
    if not tsc_errors:
        print("✅ No TS2307 (cannot find module) errors from tsc!")
        return
    
    print(f"\n📋 TSC found {len(tsc_errors)} TS2307 errors")
    
    # Build set of our detected invalid imports
    our_invalid = set()
    for imp in result.invalid_imports:
        try:
            rel_path = str(imp.file_path.relative_to(result.project_root)).replace('\\', '/')
        except ValueError:
            rel_path = str(imp.file_path).replace('\\', '/')
        our_invalid.add((rel_path, imp.line_number, imp.import_path))
    
    # Compare
    tsc_set = set(tsc_errors)
    
    # Errors tsc found but we missed
    missed = []
    for (file, line, module) in tsc_errors:
        found = False
        for (our_file, our_line, our_module) in our_invalid:
            if our_file.endswith(file) or file.endswith(our_file):
                if our_line == line and our_module == module:
                    found = True
                    break
        if not found:
            missed.append((file, line, module))
    
    if missed:
        print(f"\n⚠️  MISSED BY OUR SCANNER ({len(missed)}):")
        for file, line, module in missed[:20]:  # Show first 20
            print(f"   {file}:{line} → '{module}'")
        if len(missed) > 20:
            print(f"   ... and {len(missed) - 20} more")
    else:
        print("\n✅ We detected all TS2307 errors that tsc found!")
    
    # Extra: errors we found but tsc didn't (might be package imports we skip)
    extra = []
    for (our_file, our_line, our_module) in our_invalid:
        found = False
        for (file, line, module) in tsc_errors:
            if our_file.endswith(file) or file.endswith(our_file):
                if our_line == line and our_module == module:
                    found = True
                    break
        if not found:
            extra.append((our_file, our_line, our_module))
    
    if extra:
        print(f"\n📝 EXTRA (we found, tsc didn't - usually OK): {len(extra)}")


def debug_file_imports(file_path: Path, scanner: 'ImportScanner', project_root: Path):
    """Debug imports in a specific file with detailed resolution steps."""
    print("\n" + "=" * 80)
    print(f"DEBUG FILE: {file_path}")
    print("=" * 80)
    
    # Resolve the file path
    if not file_path.is_absolute():
        file_path = project_root / file_path
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📄 Absolute path: {file_path}")
    print(f"📁 Project root:  {project_root}")
    
    try:
        rel_path = file_path.relative_to(project_root)
        print(f"📍 Relative path: {rel_path}")
    except ValueError:
        print(f"⚠️  File is outside project root!")
    
    # Read the file and find imports
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"❌ Cannot read file: {e}")
        return
    
    lines = content.split('\n')
    print(f"\n📥 Scanning {len(lines)} lines...")
    
    # Make sure resolver is initialized
    if not scanner.resolver:
        scanner.discover_modules()
        scanner.resolver = PathResolver(
            project_root,
            scanner.tsconfig,
            scanner.modules
        )
    
    import_count = 0
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('//'):
            continue
        
        for pattern in IMPORT_PATTERNS:
            for match in pattern.finditer(line):
                import_path = match.group(2)
                import_count += 1
                
                print(f"\n{'─' * 60}")
                print(f"L{line_num}: {stripped[:80]}...")
                print(f"   Import: '{import_path}'")
                
                # Classify
                import_type = scanner.resolver.classify_import(import_path)
                print(f"   Type:   {import_type.value}")
                
                if import_type == ImportType.PACKAGE:
                    print(f"   ✅ Package import - skipped (cannot validate)")
                    continue
                
                # Resolve step by step
                from_dir = file_path.parent
                print(f"   From:   {from_dir}")
                
                if import_type == ImportType.ALIAS:
                    resolved_alias = scanner.resolver.resolve_alias(import_path)
                    print(f"   Alias resolves to: {resolved_alias}")
                    if resolved_alias:
                        target = scanner.resolver.base_url / resolved_alias
                    else:
                        print(f"   ❌ Alias not found in tsconfig.paths!")
                        continue
                elif import_type == ImportType.RELATIVE:
                    target = (from_dir / import_path).resolve()
                else:
                    target = Path(import_path)
                
                print(f"   Target: {target}")
                
                # Check if exists
                resolved = scanner.resolver._find_file(target)
                if resolved:
                    print(f"   ✅ Resolved to: {resolved}")
                else:
                    print(f"   ❌ NOT FOUND!")
                    print(f"   Tried:")
                    for ext in RESOLVE_EXTENSIONS:
                        candidate = Path(str(target) + ext)
                        exists = "✓" if candidate.exists() else "✗"
                        print(f"      {exists} {candidate}")
                    
                    # Check if directory exists
                    if target.parent.exists():
                        print(f"   📁 Parent dir exists: {target.parent}")
                        print(f"   📂 Contents: {list(target.parent.iterdir())[:5]}...")
                    else:
                        print(f"   📁 Parent dir MISSING: {target.parent}")
    
    print(f"\n{'═' * 60}")
    print(f"Total imports found: {import_count}")


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    
    parser = argparse.ArgumentParser(
        prog='fix-imports',
        description='''
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      FIX-IMPORTS v2.0 - Import Path Analyzer                     ║
║                                                                                  ║
║  Comprehensive TypeScript/JavaScript import path analyzer and fixer.            ║
║  Works with tsconfig.json, validates every import, suggests fixes.              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

BEHAVIOR:
  By default, the tool looks for tsconfig.json and uses its location as the
  project root. You can override this with explicit --tsconfig or --dir options.

  Priority order:
    1. --tsconfig FILE  → use this tsconfig, scan its include patterns
    2. --dir DIRECTORY  → scan this directory (optionally with auto-detected tsconfig)
    3. (default)        → find tsconfig.json from current dir, use as project root

EXAMPLES:
  # Auto-detect tsconfig.json and scan project
  python fix-imports.py

  # Explicitly specify tsconfig.json
  python fix-imports.py --tsconfig ./tsconfig.json

  # Scan specific directory (auto-detect tsconfig in parent dirs)
  python fix-imports.py --dir ./src_v2

  # Scan specific directory with explicit tsconfig
  python fix-imports.py --dir ./src_v2 --tsconfig ./tsconfig.json

  # Dry-run: preview HIGH confidence fixes
  python fix-imports.py --fix-high --dry-run

  # Apply HIGH confidence fixes
  python fix-imports.py --fix-high

  # Apply all fixable imports
  python fix-imports.py --fix-all

  # Export detailed report to JSON
  python fix-imports.py --export report.json --verbose

  # List all discovered modules
  python fix-imports.py --list-modules
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
ISSUE TYPES:
  🔍 cannot_resolve   - Import path doesn't resolve to any file
  📄 missing_file     - Directory exists but target file doesn't
  📎 has_extension    - Import has .ts/.tsx extension (should be omitted)
  📏 wrong_depth      - Too many or too few ../ in relative path
  🏷️  alias_not_found  - @alias not defined in tsconfig.json paths

CONFIDENCE LEVELS:
  🟢 HIGH   - Exact module match. Safe to auto-fix.
  🟡 MEDIUM - Single partial match. Review recommended.
  🟠 LOW    - Multiple matches or guess. Manual review needed.
  ⚫ NONE   - No suggestion. Manual fix required.

NOTES:
  - Reads tsconfig.json for baseUrl, paths, include, exclude
  - Package imports (lodash, @types/*) are skipped - cannot validate
  - Always use --dry-run first to preview changes
  - Backup your code before using --fix-* without --dry-run
'''
    )
    
    # Config options
    config_group = parser.add_argument_group('Target Selection')
    config_group.add_argument(
        '--tsconfig', '-c',
        type=Path,
        metavar='FILE',
        help='Path to tsconfig.json (auto-detected from current dir if not specified)'
    )
    config_group.add_argument(
        '--dir', '-d',
        type=Path,
        metavar='DIRECTORY',
        help='Directory to scan (default: tsconfig location or current dir)'
    )
    config_group.add_argument(
        '--exclude', '-e',
        action='append',
        metavar='PATTERN',
        help='Additional patterns to exclude (can use multiple times)'
    )
    config_group.add_argument(
        '--no-exclude',
        action='store_true',
        help='Ignore tsconfig exclude patterns (scan everything including BURDEL, node_modules, etc.)'
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
        help='Apply ALL suggested fixes'
    )
    fix_group.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Preview changes without applying them'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
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
        '--list-modules',
        action='store_true',
        help='List all discovered modules'
    )
    output_group.add_argument(
        '--compare-tsc',
        action='store_true',
        help='Run tsc --noEmit and compare results with our scan'
    )
    output_group.add_argument(
        '--debug-file',
        type=Path,
        metavar='FILE',
        help='Debug imports in a specific file (show detailed resolution steps)'
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
    
    # Default workspace: src_v2 relative to script location or tsconfig
    script_dir = Path(__file__).resolve().parent  # tools/
    project_root = script_dir.parent  # src3/
    default_scan_dir = project_root / 'src_v2'
    
    # Show help if no arguments and no default actions
    if len(sys.argv) == 1:
        # Try to auto-detect tsconfig
        tsconfig_path = TsConfigParser.find_tsconfig(Path.cwd())
        if not tsconfig_path:
            # Use default project structure
            tsconfig_path = project_root / 'tsconfig.json'
            if not tsconfig_path.exists():
                parser.print_help()
                print("\n❌ No tsconfig.json found. Use --tsconfig or --dir to specify target.")
                sys.exit(1)
    
    # Determine tsconfig
    tsconfig = None
    tsconfig_path = args.tsconfig
    
    if tsconfig_path:
        tsconfig_path = tsconfig_path.resolve()
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
    else:
        if not args.quiet:
            print("ℹ️  No tsconfig.json found, using defaults")
    
    # Determine directory to scan
    if args.dir:
        directory = args.dir.resolve()
        scan_dir = directory  # When --dir specified, scan that dir
    elif tsconfig:
        # Use tsconfig's directory as project root, but scan src_v2
        directory = tsconfig.config_path.parent.resolve()
        scan_dir = default_scan_dir if default_scan_dir.exists() else directory
    else:
        directory = project_root
        scan_dir = default_scan_dir if default_scan_dir.exists() else directory
    
    # Validate directory
    if not directory.exists():
        print(f"❌ Error: Directory does not exist: {directory}")
        sys.exit(1)
    if not directory.is_dir():
        print(f"❌ Error: Not a directory: {directory}")
        sys.exit(1)
    
    if not args.quiet:
        if scan_dir != directory:
            print(f"📁 Project root: {directory}")
            print(f"🔍 Scanning:     {scan_dir}")
        else:
            print(f"🔍 Scanning: {directory}")
    
    # Create scanner with separate project_root and scan_dir
    scanner = ImportScanner(
        project_root=directory,
        tsconfig=tsconfig,
        verbose=args.verbose,
        exclude_patterns=args.exclude,
        no_exclude=args.no_exclude,
        scan_dir=scan_dir
    )
    
    # Scan
    if not args.quiet:
        print(f"🔍 Scanning: {directory}")
    
    result = scanner.scan_all()
    
    # List modules if requested
    if args.list_modules:
        print("\n" + "=" * 60)
        print("DISCOVERED MODULES")
        print("=" * 60)
        for key in sorted(result.modules.keys()):
            module = result.modules[key]
            print(f"  {key} → {module.absolute_path.name}")
        print()
    
    # Debug specific file if requested
    if args.debug_file:
        debug_file_imports(args.debug_file, scanner, directory)
    
    # Compare with tsc if requested
    if args.compare_tsc:
        compare_with_tsc(directory, result)
    
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
        # Determine confidence levels
        levels: Set[Confidence] = set()
        if args.fix_high or args.fix_all:
            levels.add(Confidence.HIGH)
        if args.fix_medium or args.fix_all:
            levels.add(Confidence.MEDIUM)
        if args.fix_low or args.fix_all:
            levels.add(Confidence.LOW)
        
        # Confirm if not dry-run
        if not args.dry_run and levels:
            level_names = ', '.join(l.value.upper() for l in sorted(levels, key=lambda x: x.value))
            print(f"\n⚠️  About to apply {level_names} confidence fixes.")
            print("   Make sure you have a backup!")
            try:
                response = input("   Continue? [y/N]: ")
                if response.lower() != 'y':
                    print("Aborted.")
                    sys.exit(0)
            except EOFError:
                print("\nNo input available, aborting.")
                sys.exit(0)
        
        # Create fixer and apply
        fixer = ImportFixer(
            dry_run=args.dry_run,
            verbose=args.verbose,
            project_root=directory
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
            print("\nNo fixes to apply for selected confidence levels.")
    
    # Exit with error code if there are unfixable imports
    if result.unfixable:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    # Ensure UTF-8 output on Windows
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    main()
