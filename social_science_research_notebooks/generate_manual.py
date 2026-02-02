import nbformat
import glob
import os

def generate_manual():
    # Find all notebooks
    notebooks = sorted(glob.glob("*.ipynb"))

    manual_lines = [
        "# Social Science Research Notebooks Manual",
        "",
        "This manual is automatically generated from the notebooks in this repository.",
        "",
        "---",
        ""
    ]

    for nb_path in notebooks:
        try:
            with open(nb_path, encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            # Standard structure: Cell 2 is the Title/Description
            # (Cell 0=License, Cell 1=Setup)
            if len(nb.cells) > 2:
                cell = nb.cells[2]
                if cell.cell_type == 'markdown':
                    content = cell.source

                    # Add Notebook Filename as a kicker/header
                    manual_lines.append(f"## Notebook: `{nb_path}`")
                    manual_lines.append("")

                    # Clean up content if needed (e.g. remove redundant 'Description:')
                    # content = content.replace("**Description:**", "").strip()

                    manual_lines.append(content)
                    manual_lines.append("")
                    manual_lines.append("---")
                    manual_lines.append("")
        except Exception as e:
            print(f"Error processing {nb_path}: {e}")

    with open("MANUAL.md", "w", encoding='utf-8') as f:
        f.write("\n".join(manual_lines))

    print(f"Generated MANUAL.md from {len(notebooks)} notebooks.")

if __name__ == "__main__":
    generate_manual()
