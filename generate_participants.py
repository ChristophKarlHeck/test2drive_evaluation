import yaml

class QuotedSafeDumper(yaml.SafeDumper):
    pass

def str_as_single_quoted(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")

QuotedSafeDumper.add_representer(str, str_as_single_quoted)

if __name__ == "__main__":
    data = {"participants": [{"id": f"{i:03d}"} for i in range(1000)]}  # 000..999

    with open("participants.yaml", "w") as f:
        yaml.dump(data, f, Dumper=QuotedSafeDumper, sort_keys=False)

    print("Generated participants.yaml")