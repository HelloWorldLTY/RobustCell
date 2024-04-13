# Installation Guide for the `robustcell` Package

To install the `robustcell` package from your local system, follow these steps.

## Step 1: Navigate to Your Project Directory

Open your command line interface (CLI) and navigate to the directory where your `setup.py` file is located.

## Step 2: Install the Package Locally

You can install the `robustcell` package locally using `pip`. Point it to the directory containing `setup.py` with the following command:

```bash
pip install .
```

This command tells `pip` to install the package from the current directory (`.`). Make sure you are in the directory where `setup.py` is located when you run this command.

## Step 3: Install in Editable Mode (Development Mode)

If you hope to modify the package and want changes in the code to be immediately reflected without needing to reinstall, install it in editable mode using:

```bash
pip install -e .
```

This command installs the package in a way that allows you to modify the source code and see the changes directly without reinstallation.

## Step 4: Verify Installation

Verify that your package has been installed correctly by trying to import it in Python. Open a Python interpreter by typing `python` in your command line, and try importing your package:

```python
import robustcell
```

If there are no errors, the package has been installed successfully.

## Step 5: Uninstalling the Package

If you need to uninstall the `robustcell`, you can do so with:

```bash
pip uninstall robustcell
```
