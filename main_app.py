import os
import subprocess
import sys
sys.path.append('/Users/greenkedia/Desktop/Dream11Final/src/')
def update_data():
    # Your logic for updating data goes here
    print("Data updated successfully!")

def run_final_ui():
    # Path to Final_UI.py
    final_ui_path = "src/UI/Final_UI.py"
    subprocess.Popen(["python", final_ui_path])
    print("Final_UI.py launched!")

if __name__ == "__main__":
    update_data()
    run_final_ui()