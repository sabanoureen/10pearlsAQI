from pathlib import Path
import re

API_DIR = Path("app/api")

def fix_file(path: Path):
    text = path.read_text(encoding="utf-8")
    original = text

    text = re.sub(r"from\s+db\.", "from app.db.", text)
    text = re.sub(r"import\s+db\.", "import app.db.", text)

    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"✅ Fixed: {path}")

def main():
    print("🔧 Fixing imports in app/api...\n")
    for py in API_DIR.rglob("*.py"):
        fix_file(py)
    print("\n🎉 Done fixing API imports.")

if __name__ == "__main__":
    main()
