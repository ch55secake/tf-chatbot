import json
from datetime import datetime
from typing import Any

from model.chatbot_interface import predict_class, generate_response

intents = json.loads(open("~/resources/intents.json").read())

if __name__ == '__main__':
    print(f"Bot started running at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

    while True:
        # Read input from the user
        user_input: str = input()
        intentions: list[dict[str, str | Any]] = predict_class(user_input=user_input)
        # Generate the response and push to user in below print
        response: str = generate_response(list_of_intents=intentions)
        print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Bot generated the response of:\n{response}")