import sys
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import os
import argparse

def verify_notebook(path, execute=True):
    print(f"Verifying {path}...")
    try:
        with open(path, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Validate structure
        nbformat.validate(nb)

        if execute:
            # Execute
            # We assume the environment has dependencies installed.
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

            # Determine path to run in
            run_path = os.path.dirname(path)
            if run_path == "":
                run_path = "."

            ep.preprocess(nb, {'metadata': {'path': run_path}})
            print(f"SUCCESS (Executed): {path}")
        else:
             print(f"SUCCESS (Validated): {path}")

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
    parser = argparse.ArgumentParser(description='Verify Jupyter Notebooks')
    parser.add_argument('files', metavar='F', type=str, nargs='*', help='notebook files to verify')
    parser.add_argument('--no-exec', action='store_true', help='Do not execute notebooks, only validate structure')

    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        # Find all notebooks in current directory
        # Change to use os.walk or explicitly look in social_science_research_notebooks/
        # Since this script is likely run from root, we should look in the subdirectory
        files = sorted(glob.glob("social_science_research_notebooks/*.ipynb"))

    print(f"Found {len(files)} notebooks to verify.")

    failures = []
    for f in files:
        if not verify_notebook(f, execute=not args.no_exec):
            failures.append(f)

    if failures:
        print(f"\n{len(failures)} Notebooks Failed:")
        for f in failures:
            print(f"- {f}")
        sys.exit(1)
    else:
        print("\nAll notebooks passed successfully.")
