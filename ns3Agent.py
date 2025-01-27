import subprocess

def detect_language(code):
    python_indicators = ['def ', 'import ', 'print(', 'for ', 'in ', 'if ', 'from ns import ns']
    cpp_indicators = ['#include', 'int main()', 'using namespace ', ';']

    python_score = sum(1 for indicator in python_indicators if indicator in code)
    cpp_score = sum(1 for indicator in cpp_indicators if indicator in code)

    if python_score > cpp_score:
        return 'python'
    else:
        return 'cpp'

def run_ns3_simulation(code):
    language = detect_language(code)
    if language == 'python':
        return run_python_simulation(code)
    elif language == 'cpp':
        return run_cpp_simulation(code)
    else:
        return "Unable to determine the language of the provided code."

def run_python_simulation(python_code):
    # Add __file__ simulation
    simulated_file_code = """
import os
import sys
if not hasattr(sys, 'argv'):
    sys.argv  = ['']
__file__ = os.path.abspath('<simulated_file>')
""" + python_code

    # Execute the filtered code
    try:
        result = subprocess.run(["python", "-c", simulated_file_code], capture_output=True, text=True)
        combined_output = result.stdout + result.stderr
        filtered_output = '\n'.join(
            line for line in combined_output.split('\n') 
            if not ("Fail" in line or "Warning" in line or "warning" in line or "fail" in line)
        )
        print(f"{filtered_output}")
        return f"{filtered_output}"
    except Exception as e:
        return f"An error occurred during execution: {e}"

def run_cpp_simulation(cpp_code):
    ns3_file_path = "ns-3-dev/scratch/GenOnet.cc"
    
    
    # Step 1: Clear the content of GenOnet.cc
    try:
        with open(ns3_file_path, 'w') as file:
            file.write("")
    except Exception as e:
        return f"An error occurred while clearing the file: {e}"
    
    # Step 2: Write the new C++ code to GenOnet.cc
    try:
        with open(ns3_file_path, 'w') as file:
            file.write(cpp_code)
    except Exception as e:
        return f"An error occurred while writing to the file: {e}"
    
    # Step 3: Run the ns-3 simulation
    try:
        result = subprocess.run(["./ns3", "run", "GenOnet"], cwd="ns-3-dev", capture_output=True, text=True)
        combined_output = result.stdout + result.stderr
        filtered_output = '\n'.join(
            line for line in combined_output.split('\n') 
            if not ("Fail" in line or "Warning" in line or "warning" in line or "fail" in line)
        )
        print(f"{filtered_output}")
        return f"{filtered_output}"
    except Exception as e:
        return f"An error occurred during execution: {e}"

