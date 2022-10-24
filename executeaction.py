import webbrowser

YOUTUBE_URL = "https://www.youtube.com/feed/subscriptions"

def execute_action(finger_count):
    if finger_count == 5:
        print("Open Youtube")
        webbrowser.open(YOUTUBE_URL)
    elif finger_count == 4:
        print("ACTION 4")
    elif finger_count == 3:
        print("ACTION 3")
    elif finger_count == 2:
        print("ACTION 2")
    elif finger_count == 1:
        print("ACTION 1")
    elif finger_count == 0:
        print("ACTION 0")
