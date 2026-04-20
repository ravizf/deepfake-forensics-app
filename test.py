import requests


URL = "http://127.0.0.1:10000/api/analyze"


def main():
    with open("test.jpg", "rb") as sample:
        response = requests.post(URL, files={"file": sample})
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
