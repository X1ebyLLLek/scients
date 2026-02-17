import zipfile
import os

def create_package(zip_name="scients_package.zip"):
    # Files to include (source code + config)
    include_extensions = ['.py', '.txt', '.ipynb', '.md', '.csv']
    exclude_dirs = ['__pycache__', '.git', '.idea', 'venv', '.ipynb_checkpoints', 'runs']
    exclude_files = [zip_name, 'create_package.py', 'transformer_ueba_model.pth', 
                     'checkpoint.pth', 'BGL.log_structured.csv', 'performance.png',
                     'comparison.png', 'report_text.txt']  # Large artifacts

    print(f"Creating {zip_name}...")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through current directory
        for root, dirs, files in os.walk("."):
            # Exclude directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                
                # Check extension
                _, ext = os.path.splitext(file)
                if ext in include_extensions or file == 'requirements.txt':
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, ".")
                    print(f"Adding: {arcname}")
                    zipf.write(file_path, arcname)
                    
    print(f"\nSuccessfully created {zip_name}!")
    print(f"Size: {os.path.getsize(zip_name) / 1024:.2f} KB")

if __name__ == "__main__":
    create_package()
