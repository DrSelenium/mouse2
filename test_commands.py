#!/usr/bin/env python3
"""
Command Tester for Micromouse API
This script loads the generated test commands and can be used to test them against a running API.
"""

import json
import requests
from typing import List, Dict, Any

class CommandTester:
    """Test the generated commands against the micromouse API."""

    def __init__(self, base_url: str = "http://localhost:8000", commands_file: str = "test_commands.json"):
        self.base_url = base_url
        self.commands_file = commands_file
        self.test_commands = self.load_commands()

    def load_commands(self) -> List[Dict[str, Any]]:
        """Load test commands from JSON file."""
        try:
            with open(self.commands_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Commands file {self.commands_file} not found. Run test_generate_commands.py first.")
            return []

    def test_single_command(self, command: Dict[str, Any], initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test a single command sequence against the API."""
        if initial_state is None:
            initial_state = {
                "game_uuid": "test-uuid-123",
                "sensor_data": [12, 12, 12, 12, 12],  # Clear path ahead
                "total_time_ms": 0,
                "goal_reached": False,
                "best_time_ms": None,
                "run_time_ms": 0,
                "run": 0,
                "momentum": 0,
                "instructions": command["instructions"],  # Send instructions to execute
                "end": False
            }
        else:
            initial_state["instructions"] = command["instructions"]
            initial_state["end"] = False

        try:
            response = requests.post(
                f"{self.base_url}/micro-mouse",
                json=initial_state,
                timeout=10
            )

            result = {
                'command_name': command['name'],
                'description': command['description'],
                'instructions_sent': command['instructions'],
                'status_code': response.status_code,
                'success': response.status_code == 200
            }

            if response.status_code == 200:
                result['response'] = response.json()
            else:
                result['error'] = response.text

            return result

        except requests.exceptions.RequestException as e:
            return {
                'command_name': command['name'],
                'description': command['description'],
                'instructions_sent': command['instructions'],
                'error': str(e),
                'success': False
            }

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all test commands."""
        results = []
        for command in self.test_commands:
            result = self.test_single_command(command)
            results.append(result)
            status = "PASS" if result['success'] else "FAIL"
            print(f"Test: {command['name']} - {status}")
            if not result['success']:
                print(f"  Error: {result.get('error', 'Unknown error')}")
        return results

    def run_test_by_name(self, test_name: str) -> Dict[str, Any]:
        """Run a specific test by name."""
        for command in self.test_commands:
            if command['name'] == test_name:
                return self.test_single_command(command)
        return {'error': f'Test {test_name} not found'}

    def list_available_tests(self):
        """List all available test commands."""
        print("Available test commands:")
        for i, cmd in enumerate(self.test_commands, 1):
            print(f"{i}. {cmd['name']}: {cmd['description']}")
            print(f"   Instructions: {cmd['instructions']}")

def main():
    tester = CommandTester()

    if not tester.test_commands:
        print("No test commands found. Please run test_generate_commands.py first.")
        return

    print(f"Loaded {len(tester.test_commands)} test commands.")
    print("\nTo run tests:")
    print("1. Start the micromouse API server: python main.py")
    print("2. Run: python test_commands.py --run-all")
    print("3. Or run specific test: python test_commands.py --test accelerate_from_rest")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        tester = CommandTester()

        if sys.argv[1] == "--run-all":
            results = tester.run_all_tests()
            # Save results
            with open("command_test_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print("\nResults saved to command_test_results.json")

        elif sys.argv[1] == "--list":
            tester.list_available_tests()

        elif sys.argv[1] == "--test" and len(sys.argv) > 2:
            result = tester.run_test_by_name(sys.argv[2])
            print(json.dumps(result, indent=2))

        else:
            print("Usage:")
            print("  python test_commands.py --run-all    # Run all tests")
            print("  python test_commands.py --list       # List available tests")
            print("  python test_commands.py --test <name> # Run specific test")
    else:
        main()
