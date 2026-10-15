#!/usr/bin/env python3
"""CLI utility for LMTrust Agent Bridge (Zero-API testing request & apply)."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bridge.lmtrust_bridge import (
    create_test_request,
    list_pending_requests,
    get_test_response,
    apply_test_response,
)


def main():
    parser = argparse.ArgumentParser(description="LMTrust Local Agent Bridge CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # request
    req_parser = subparsers.add_parser("request", help="Submit a blind-spot test request")
    req_parser.add_argument("source_file", help="Path to the source file to analyze")
    req_parser.add_argument("--layers", default="L1,L2,L4,L4b,L7,L9", help="Comma-separated layers")
    req_parser.add_argument("--note", default="", help="Optional instructions")

    # status
    subparsers.add_parser("status", help="List pending requests")

    # apply
    apply_parser = subparsers.add_parser("apply", help="Apply generated tests from outbox")
    apply_parser.add_argument("request_id", help="ID of the request to apply")
    apply_parser.add_argument("--target", default=None, help="Custom target test file path")

    args = parser.parse_args()

    if args.command == "request":
        try:
            req = create_test_request(args.source_file, focus_layers=args.layers, note=args.note)
            print(f"✅ Created LMTrust request: {req['id']}")
            print(f"   Source file: {req['source_file']}")
            print(f"   Focus layers: {', '.join(req['focus_layers'])}")
            print("   Pending Antigravity / Agent processing in .argent/lmtrust_bridge/inbox/")
        except Exception as e:
            print(f"❌ Error creating request: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "status":
        pending = list_pending_requests()
        if not pending:
            print("No pending requests in .argent/lmtrust_bridge/inbox/")
        else:
            print(f"Pending LMTrust requests ({len(pending)}):")
            for req in pending:
                print(f"- {req['id']}: {req['source_file']} (Layers: {', '.join(req['focus_layers'])})")

    elif args.command == "apply":
        res = apply_test_response(args.request_id, target_file=args.target)
        print(res)


if __name__ == "__main__":
    main()
