from pathlib import Path
import re

PIPELINES_DIR = Path("app/pipelines")

def fix_file(path: Path):
    text = path.read_text(encoding="utf-8")
    original = text

    # Replace bad imports
    text = re.sub(r"from\s+pipelines\.", "from app.pipelines.", text)
    text = re.sub(r"import\s+pipelines\.", "import app.pipelines.", text)

    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"✅ Fixed: {path}")

def main():
    print("🔧 Fixing imports in app/pipelines...\n")

    for py_file in PIPELINES_DIR.rglob("*.py"):
        fix_file(py_file)

    print("\n🎉 Done fixing pipeline imports.")

if __name__ == "__main__":
    main()
