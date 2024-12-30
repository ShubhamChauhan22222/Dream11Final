import tkinter as tk
import subprocess
import webbrowser

import sys
sys.path.append('/Users/greenkedia/Desktop/Dream11Final/src/')

def run_product_ui():
    google_drive_link = "https://drive.google.com/drive/folders/16eCHg7xAsPVll448r5HHdUSSztjsxh61?usp=sharing"
    webbrowser.open(google_drive_link)
    print("Google Drive link opened!")

def run_model_ui():
    model_ui_path = "src\\UI\\Model_UI.py"
    subprocess.Popen(["streamlit","run", model_ui_path])
    print("Model_UI.py launched!")

# Tkinter UI setup
def main():
    root = tk.Tk()
    root.title("Final UI")

    tk.Label(root, text="Choose an option:", font=("Arial", 14)).pack(pady=10)

    btn_product_ui = tk.Button(root, text="Run Product_UI", command=run_product_ui, width=20, bg="lightblue")
    btn_product_ui.pack(pady=5)

    btn_model_ui = tk.Button(root, text="Run Model_UI", command=run_model_ui, width=20, bg="lightgreen")
    btn_model_ui.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()