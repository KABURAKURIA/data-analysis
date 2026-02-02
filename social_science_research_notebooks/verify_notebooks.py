import sys
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import os

def verify_notebook(path):
    print(f"Verifying {path}...")
    try:
        with open(path, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Validate structure
        nbformat.validate(nb)

        # Execute
        # We assume the environment has dependencies installed.
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

        # Determine path to run in
        run_path = os.path.dirname(path)
        if run_path == "":
            run_path = "."

        ep.preprocess(nb, {'metadata': {'path': run_path}})
        print(f"SUCCESS: {path}")
        return True
    except CellExecutionError as e:
        print(f"ERROR executing {path}:")
        # Print only the last part of the traceback to keep logs clean
        print(str(e).split('\n')[-5:])
        return False
    except Exception as e:
        print(f"ERROR validating {path}:")
        print(e)
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Find all notebooks in current directory
        files = sorted(glob.glob("*.ipynb"))

    print(f"Found {len(files)} notebooks to verify.")

    failures = []
    for f in files:
        if not verify_notebook(f):
            failures.append(f)

    if failures:
        print(f"\n{len(failures)} Notebooks Failed:")
        for f in failures:
            print(f"- {f}")
        sys.exit(1)
    else:
        print("\nAll notebooks passed successfully.")
