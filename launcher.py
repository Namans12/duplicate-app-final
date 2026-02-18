#!/usr/bin/env python3
"""
Duplicate File Finder Launcher
Easy launcher for choosing between different versions
"""
import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import PyQt6
        return True, "PyQt6 available"
    except ImportError:
        return False, "PyQt6 not found. Install with: pip install PyQt6"

def check_enterprise_modules():
    """Check if enterprise modules are available"""
    missing = []
    
    modules = [
        "database_manager.py",
        "gpu_accelerator.py", 
        "plugin_system.py",
        "memory_optimizer.py",
        "duplicate_merger.py"
    ]
    
    for module in modules:
        if not os.path.exists(module):
            missing.append(module)
            
    return len(missing) == 0, missing

def launch_application(script_name):
    """Launch the specified application"""
    try:
        print(f"🚀 Launching {script_name}...")
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching {script_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ {script_name} interrupted by user")
        return False
    return True

def main():
    """Main launcher interface"""
    print("=" * 60)
    print("🔍 DUPLICATE FILE FINDER - VERSION LAUNCHER")
    print("=" * 60)
    
    # Check basic dependencies
    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        print(f"❌ {deps_msg}")
        print("Please install dependencies first:")
        print("pip install -r requirements.txt")
        return
    
    print(f"✅ {deps_msg}")
    
    # Check enterprise modules
    enterprise_ok, missing_modules = check_enterprise_modules()
    
    print("\n📋 AVAILABLE VERSIONS:")
    print("-" * 30)
    
    options = []
    
    # Basic version
    if os.path.exists("main.py"):
        options.append(("1", "Basic Duplicate Finder", "main.py", "Simple duplicate detection"))
        print("1️⃣  Basic Duplicate Finder (main.py)")
        print("    ➤ Simple duplicate detection")
        print("    ➤ Basic PyQt6 interface")
        print()
    
    # Enhanced UX version
    if os.path.exists("main_enhanced.py"):
        options.append(("2", "Enhanced UX Version", "main_enhanced.py", "All UX improvements"))
        print("2️⃣  Enhanced UX Version (main_enhanced.py) ⭐ RECOMMENDED")
        print("    ➤ Search within duplicates")
        print("    ➤ Progress details with ETA")
        print("    ➤ Dark/Light theme toggle")
        print("    ➤ Keyboard shortcuts")
        print("    ➤ Professional interface")
        print()
    
    # Enterprise version
    if os.path.exists("main_enterprise.py"):
        if enterprise_ok:
            options.append(("3", "Enterprise Edition", "main_enterprise.py", "Full enterprise features"))
            print("3️⃣  Enterprise Edition (main_enterprise.py) 🚀 ADVANCED")
            print("    ➤ All UX enhancements")
            print("    ➤ Database persistence")
            print("    ➤ GPU acceleration")
            print("    ➤ Plugin system")
            print("    ➤ Memory optimization")
            print("    ➤ File merging")
            print()
        else:
            print("3️⃣  Enterprise Edition (main_enterprise.py) ⚠️  UNAVAILABLE")
            print("    ❌ Missing enterprise modules:")
            for module in missing_modules:
                print(f"       - {module}")
            print()
    
    if not options:
        print("❌ No application versions found!")
        return
    
    # Get user choice
    print("🎯 SELECT VERSION TO LAUNCH:")
    print("-" * 30)
    
    while True:
        try:
            choice = input(f"Enter choice (1-{len(options)}) or 'q' to quit: ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                return
                
            # Find matching option
            selected_option = None
            for opt in options:
                if opt[0] == choice:
                    selected_option = opt
                    break
                    
            if selected_option:
                name, description, script, details = selected_option
                print(f"\n🎯 You selected: {description}")
                print(f"📝 Features: {details}")
                
                confirm = input("Launch this version? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    success = launch_application(script)
                    if success:
                        print(f"✅ {description} closed successfully")
                    break
                else:
                    print("❌ Launch cancelled\n")
                    continue
            else:
                print(f"❌ Invalid choice '{choice}'. Please try again.\n")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
