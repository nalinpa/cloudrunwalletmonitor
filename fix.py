import os

def remove_bom_from_env_file():
    """Remove BOM from .env.production file"""
    
    filename = '.env.production'
    backup_filename = '.env.production.backup'
    
    print("🔧 Removing BOM from .env.production...")
    
    # Read the file
    with open(filename, 'r', encoding='utf-8-sig') as f:  # utf-8-sig auto-removes BOM
        content = f.read()
    
    print(f"✅ Read file: {len(content)} characters")
    
    # Backup original
    with open(backup_filename, 'w', encoding='utf-8') as f:
        with open(filename, 'r', encoding='utf-8') as original:
            f.write(original.read())
    print(f"💾 Created backup: {backup_filename}")
    
    # Write without BOM
    with open(filename, 'w', encoding='utf-8') as f:  # utf-8 without BOM
        f.write(content)
    
    print(f"✅ Fixed file written without BOM")
    
    # Verify the fix
    print("\n🧪 Verification:")
    with open(filename, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        print(f"First line: {repr(first_line)}")
        
        if first_line.startswith('MONGO_URI='):
            print("✅ BOM successfully removed!")
            return True
        else:
            print("❌ BOM still present or other issue")
            return False

if __name__ == "__main__":
    success = remove_bom_from_env_file()
    
    if success:
        print("\n🎯 Now test your environment loading:")
        print("python debug.py")
    else:
        print("\n❌ Manual fix required. Use the clean .env.production content provided.")