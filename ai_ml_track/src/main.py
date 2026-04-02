import os
import json
from grouper import grouped_output, refine_output

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, "data", "sample_input.json")
    with open(input_path) as f:
        texts = json.load(f)

    
    groups, test_length = grouped_output(texts)
    final_output = refine_output(groups, test_length)
    print(final_output)


if __name__ == "__main__":
    main()

    