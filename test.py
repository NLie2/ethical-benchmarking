# * test an AI model on a set of questions (INCLUDE FUNCTIONALITY TO CHANGE CONTEXTS)

# example.py
import sys

def function1(arg):
    print(f"Function 1 called with argument {arg}")

def function2(arg):
    print(f"Function 2 called with argument {arg}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        function_name = sys.argv[1]
        if function_name == "function1" and len(sys.argv) == 3:
            function1(sys.argv[2])
        elif function_name == "function2" and len(sys.argv) == 3:
            function2(sys.argv[2])
        else:
            print("Invalid function name or arguments")
    else:
        print("No function name provided")
