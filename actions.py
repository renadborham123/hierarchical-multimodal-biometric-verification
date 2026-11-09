"""Simulated output only. The authorized pipeline is the only production caller."""


def execute_command(command):
    messages = {
        "STATUS": "Simulated system status: online; no physical devices connected.",
        "REPORT": "Sample report: 4 enrolled employees; simulated attendance: 3 present.",
        "UNLOCK": "Simulated protected access unlocked",
        "LOGOUT": "Verified session ended.",
    }
    if command not in messages:
        raise ValueError("Unsupported command")
    print(messages[command])
    return {"message": messages[command], "session_ended": command == "LOGOUT"}
