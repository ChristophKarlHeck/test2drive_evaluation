import random
import yaml

def generate_codes(n):
    codes = set()
    while len(codes) < n:
        code = f"{random.randint(0, 9999):04d}"
        if code != "0000":  # reserve admin
            codes.add(code)
    return sorted(list(codes))

if __name__ == "__main__":
    NUM_PARTICIPANTS = 50

    data = {
        "participants": [{"id": "0000"}] + [
            {"id": code} for code in generate_codes(NUM_PARTICIPANTS)
        ]
    }

    with open("participants.yaml", "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print("Generated participants.yaml")
