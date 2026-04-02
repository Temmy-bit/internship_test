import os
import json
from src.grouper import grouped_output, refine_output


if __name__ == "__main__":
    import json

    test_texts = [
        "Uber trip 1200",
        "UBER EATS ORDER 3400",
        "uber ride lagos 1100",
        "Netflix subscription 4500",
        "NETFLIX.COM 4500",
        "Amazon Web Services invoice",
        "AWS charges July",
        "Bolt ride 900",
        "BOLT TECHNOLOGIES 1050",
        "Paystack transfer fee",
        "Flutterwave payout 15000",
        "MTN airtime recharge 500",
        "MTN data bundle 1200",
        "Airtel subscription 800",
        "Shoprite purchase 3200",
        "Shoprite Lagos 2800",
    ]


def main():
    base_dir = os.path.join("C:\\Users\\HP\\Desktop", "internship_test", "ai_ml_track", "data")
    input_path = os.path.join(base_dir, "sample_input.json")
    with open(input_path) as f:
        texts = json.load(f)

    
    groups, test_length = grouped_output(texts)
    final_output = refine_output(groups, test_length)
    print(final_output)


if __name__ == "__main__":
    main()

    