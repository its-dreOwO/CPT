import argparse
import sys
import json
import httpx

# In the future, this could hit specific endpoints.
# For now, it interacts via REST API or modifies local configuration states (e.g., Redis).
SERVER_URL = "http://localhost:8000"


def get_status():
    """Fetches the current status of the CPT server."""
    try:
        response = httpx.get(f"{SERVER_URL}/status", timeout=5.0)
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error fetching status: {e}")
        print("Make sure the CPT server is running.")


def toggle_notification(channel: str, state: str):
    """
    Toggles a notification channel (discord, whatsapp, zalo) on or off.
    Could send an HTTP request to the server, or in the future, set a Redis flag.
    """
    # For now, we will mock the server endpoint for demonstration
    # Future implementation will hit an endpoint like POST /api/config/notifications
    print(f"[*] Setting notification toggle for '{channel}' to '{state}'...")
    try:
        # Example: httpx.post(f"{SERVER_URL}/api/config/notifications", json={"channel": channel, "enabled": state == "on"})
        print(f"✅ Notification channel '{channel}' is now {state.upper()}.")
    except Exception as e:
        print(f"❌ Failed to toggle notification channel: {e}")


def main():
    parser = argparse.ArgumentParser(description="CPT (Crypto Price Tracker) Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status Command
    subparsers.add_parser("status", help="Get the current status of all engines and models")

    # Notifications Command
    notify_parser = subparsers.add_parser("notify", help="Manage notification channels")
    notify_parser.add_parser("list", help="List all notification channels and their status")

    toggle_parser = notify_parser.add_subparsers(
        dest="toggle_cmd", help="Toggle a notification channel"
    )
    enable_parser = toggle_parser.add_parser("enable", help="Enable a channel")
    enable_parser.add_argument(
        "channel", choices=["discord", "whatsapp", "zalo", "all"], help="Channel to enable"
    )

    disable_parser = toggle_parser.add_parser("disable", help="Disable a channel")
    disable_parser.add_argument(
        "channel", choices=["discord", "whatsapp", "zalo", "all"], help="Channel to disable"
    )

    # Predictions Command
    pred_parser = subparsers.add_parser("predict", help="Get latest predictions")
    pred_parser.add_argument("coin", choices=["SOL", "DOGE"], help="Coin to predict")

    args = parser.parse_args()

    if args.command == "status":
        get_status()
    elif args.command == "notify":
        if args.toggle_cmd == "enable":
            toggle_notification(args.channel, "on")
        elif args.toggle_cmd == "disable":
            toggle_notification(args.channel, "off")
        elif args.toggle_cmd == "list":
            print("Current Notification States (Mock):")
            print(" - Discord: ON")
            print(" - WhatsApp: OFF")
            print(" - Zalo: OFF")
        else:
            notify_parser.print_help()
    elif args.command == "predict":
        try:
            response = httpx.get(f"{SERVER_URL}/predictions/{args.coin}", timeout=5.0)
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
        except Exception as e:
            print(f"Error fetching prediction: {e}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
