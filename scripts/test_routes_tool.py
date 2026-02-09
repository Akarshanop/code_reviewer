#!/usr/bin/env python3
"""Test the list_api_routes tool logic directly"""
import os
import re

base_dir = "/Users/akarshan.tyagi/Desktop/Code_reviewer/backend/data/repos"
routes = []

# Patterns for different frameworks
route_patterns = [
    # Express.js patterns
    (r"router\.(get|post|put|delete|patch)\s*\(['\"]([^'\"]+)['\"]", "express"),
    (r"app\.(get|post|put|delete|patch)\s*\(['\"]([^'\"]+)['\"]", "express"),
]

for root, dirs, files in os.walk(base_dir):
    dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__', 'dist', 'build']]
    
    for filename in files:
        if not filename.endswith(('.js', '.ts', '.py', '.java')):
            continue
        
        filepath = os.path.join(root, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    for pattern, framework in route_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            groups = match.groups()
                            method = groups[0].upper()
                            path = groups[1]
                            
                            rel_path = filepath.replace(base_dir + '/', '')
                            
                            # Get code snippet
                            start = max(0, i)
                            end = min(len(lines), i + 5)
                            code_snippet = '\n'.join(lines[start:end])
                            
                            routes.append({
                                "method": method,
                                "path": path,
                                "file": filename,
                                "file_path": rel_path,
                                "line_number": i + 1,
                                "code_snippet": code_snippet
                            })
        except Exception:
            continue

print(f"Total routes found: {len(routes)}")
print()

# Group by method
by_method = {}
for route in routes:
    method = route["method"]
    if method not in by_method:
        by_method[method] = []
    by_method[method].append(route)

print(f"Routes by method: {{{', '.join(f'{k}: {len(v)}' for k, v in by_method.items())}}}")
print()

print("=== Sample Routes ===")
for route in routes[:10]:
    print(f"\n{route['method']} {route['path']}")
    print(f"  File: {route['file_path']}:{route['line_number']}")
    print(f"  Code:")
    for line in route['code_snippet'].split('\n')[:3]:
        print(f"    {line}")
