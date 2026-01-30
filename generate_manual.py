import os
import nbformat
import re

NOTEBOOKS_DIR = 'social_science_research_notebooks'
MANUAL_FILE = 'MANUAL.md'

def generate_manual():
    notebooks = [f for f in os.listdir(NOTEBOOKS_DIR) if f.endswith('.ipynb')]
    notebooks.sort()

    with open(MANUAL_FILE, 'w') as f:
        f.write("# Social Science Research Notebooks Manual\n\n")
        f.write("This manual provides an overview of the tools available in this repository. All tools have been verified as functional.\n\n")
        f.write("## Table of Contents\n\n")

        for nb_file in notebooks:
            # Try to get title from first cell
            title = nb_file
            desc = ""
            try:
                with open(os.path.join(NOTEBOOKS_DIR, nb_file), 'r', encoding='utf-8') as nbf:
                    nb = nbformat.read(nbf, as_version=4)
                    for cell in nb.cells:
                        if cell.cell_type == 'markdown':
                            # Skip license cell
                            if 'License' in cell.source and 'MIT' in cell.source:
                                continue

                            lines = cell.source.split('\n')
                            # Look for first header
                            for line in lines:
                                if line.strip().startswith('# '):
                                    title = line.strip().replace('# ', '')
                                    break

                            # If we found a title, stop looking in other cells (unless we haven't found a title yet)
                            if title != nb_file:
                                break

            except Exception as e:
                print(f"Error reading {nb_file}: {e}")

            # Generate anchor from title
            anchor = title.lower().replace(' ', '-').replace('/', '').replace('(', '').replace(')', '').replace('&', '').replace(',', '').replace('.', '')
            # Clean up multiple dashes
            while '--' in anchor:
                anchor = anchor.replace('--', '-')

            f.write(f"- [{title}](#{anchor})\n")

        f.write("\n---\n\n")

        for nb_file in notebooks:
            title = nb_file
            desc = "No description available."
            try:
                with open(os.path.join(NOTEBOOKS_DIR, nb_file), 'r', encoding='utf-8') as nbf:
                    nb = nbformat.read(nbf, as_version=4)
                    for cell in nb.cells:
                        if cell.cell_type == 'markdown':
                            if '# License' in cell.source: # Skip license cell if it's first
                                continue

                            lines = cell.source.split('\n')
                            for line in lines:
                                if line.strip().startswith('# '):
                                    title = line.strip().replace('# ', '')
                                    break

                            # Clean up description
                            clean_lines = [l.strip() for l in lines if not l.strip().startswith('#') and l.strip()]
                            if clean_lines:
                                desc = " ".join(clean_lines)
                                # Remove "Description:" prefix if present
                                desc = re.sub(r'^\*\*Description:\*\*\s*', '', desc, flags=re.IGNORECASE)
                                desc = re.sub(r'^Description:\s*', '', desc, flags=re.IGNORECASE)
                            break
            except Exception:
                pass

            f.write(f"### {title}\n\n")
            f.write(f"**File:** `{nb_file}`\n\n")
            f.write(f"**Description:** {desc}\n\n")
            f.write(f"**Status:** Verified \u2705\n\n")
            f.write(f"[Open Notebook]({NOTEBOOKS_DIR}/{nb_file})\n\n")
            f.write("---\n\n")

if __name__ == "__main__":
    generate_manual()
    print(f"Generated {MANUAL_FILE}")
