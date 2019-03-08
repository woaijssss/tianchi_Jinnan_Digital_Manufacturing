
import json

if __name__ == '__main__':
    with open("../src/submit.json", "r+") as fd:
        lines = fd.read()
        json_str = json.loads(lines)
        results = json_str['results']
        print(len(results))