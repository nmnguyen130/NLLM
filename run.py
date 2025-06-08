import os
import time

MODULES = {
    # Tokenizers
    '1': 'src.shared.tokenizers.main',
    '2': 'src.shared.tokenizers.test',
    '3': 'src.shared.tokenizers.gpt4',
}

def print_modules():
    print("\nAvailable modules:")
    print("-" * 50)
    for num, module in MODULES.items():
        name = module.split('.')[-1].replace('_', ' ').title()
        print(f"{num}. {name} ({module})")
    print("-" * 50)

def run_module(choice):
    if choice in MODULES:
        command = f"python -m {MODULES[choice]}"
        print(f"\nExecuting: {command}")
        os.system(command)
    else:
        print("Invalid choice! Please try again.")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
    time.sleep(0.1)

def main():
    while True:
        print_modules()
        choice = input("\nEnter the number of the module to run (0 to exit): ")
        
        if choice == '0':
            print("Exiting...")
            break
            
        run_module(choice)
        
        print("\nPress any key to exit or Enter to continue...")
        key = input()
        if key:
            break
        clear_screen()

if __name__ == "__main__":
    main() 