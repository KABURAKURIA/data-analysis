import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError
import sys

NOTEBOOKS_DIR = 'social_science_research_notebooks'

def verify_notebooks():
    results = []

    # List all .ipynb files
    notebooks = [f for f in os.listdir(NOTEBOOKS_DIR) if f.endswith('.ipynb')]
    notebooks.sort()

    print(f"Found {len(notebooks)} notebooks. Starting verification...")

    for notebook_filename in notebooks:
        notebook_path = os.path.join(NOTEBOOKS_DIR, notebook_filename)
        print(f"Verifying {notebook_filename}...", end=' ', flush=True)

        try:
            with open(notebook_path) as f:
                nb = nbformat.read(f, as_version=4)

            ep = ExecutePreprocessor(timeout=60, kernel_name='python3')

            # Execute the notebook
            # We execute in the notebook's directory so relative paths work
            ep.preprocess(nb, {'metadata': {'path': NOTEBOOKS_DIR}})

            print("PASS")
            results.append({'name': notebook_filename, 'status': 'PASS', 'error': None})

        except CellExecutionError as e:
            print("FAIL")
            print(f"Error in {notebook_filename}: {e}")
            results.append({'name': notebook_filename, 'status': 'FAIL', 'error': str(e)})
        except Exception as e:
            print("FAIL")
            print(f"Error executing {notebook_filename}: {e}")
            results.append({'name': notebook_filename, 'status': 'FAIL', 'error': str(e)})

    return results

def generate_report(results):
    with open('VERIFICATION_REPORT.md', 'w') as f:
        f.write("# Notebook Verification Report\n\n")
        f.write("| Notebook | Status | Error |\n")
        f.write("| --- | --- | --- |\n")
        for res in results:
            error_msg = res['error'] if res['error'] else ""
            # Truncate error message if too long
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            error_msg = error_msg.replace("\n", " ")
            f.write(f"| {res['name']} | {res['status']} | {error_msg} |\n")

if __name__ == "__main__":
    results = verify_notebooks()
    generate_report(results)

    # Exit with code 1 if any failures
    if any(r['status'] == 'FAIL' for r in results):
        sys.exit(1)
    sys.exit(0)
