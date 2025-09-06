#!/usr/bin/env python3
"""
Test script to generate commands for testing the Micromouse API.
This script generates various sequences of movement tokens to test different scenarios.
"""

import json
import requests
from typing import List, Dict, Any
import random

# Movement tokens from the requirements
BASIC_LONGITUDINAL = ['F0', 'F1', 'F2', 'BB', 'V0', 'V1', 'V2']
IN_PLACE_ROTATIONS = ['L', 'R']
MOVING_ROTATIONS = [
    'F0L', 'F0R', 'F1L', 'F1R', 'F2L', 'F2R',
    'BBL', 'BBR', 'V0L', 'V0R', 'V1L', 'V1R', 'V2L', 'V2R'
]
CORNER_TURNS = [
    'F0LT', 'F0LW', 'F0RT', 'F0RW',
    'F1LT', 'F1LW', 'F1RT', 'F1RW',
    'F2LT', 'F2LW', 'F2RT', 'F2RW',
    'V0LT', 'V0LW', 'V0RT', 'V0RW',
    'V1LT', 'V1LW', 'V1RT', 'V1RW',
    'V2LT', 'V2LW', 'V2RT', 'V2RW'
]

ALL_TOKENS = BASIC_LONGITUDINAL + IN_PLACE_ROTATIONS + MOVING_ROTATIONS + CORNER_TURNS

class CommandGenerator:
    """Generates test command sequences for micromouse testing."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_cases = []

    def generate_basic_movement_tests(self) -> List[Dict[str, Any]]:
        """Generate test cases for basic longitudinal movements."""
        tests = []

        # Test acceleration from rest
        tests.append({
            'name': 'accelerate_from_rest',
            'instructions': ['F1', 'F2', 'F2', 'BB'],
            'description': 'Accelerate from 0 to 4, then brake'
        })

        # Test deceleration
        tests.append({
            'name': 'decelerate_sequence',
            'instructions': ['F2', 'F2', 'F1', 'F0', 'F0'],
            'description': 'Accelerate then decelerate to 0'
        })

        # Test reverse movements
        tests.append({
            'name': 'reverse_movements',
            'instructions': ['V1', 'V2', 'V2', 'BB'],
            'description': 'Reverse accelerate and brake'
        })

        return tests

    def generate_rotation_tests(self) -> List[Dict[str, Any]]:
        """Generate test cases for rotation commands."""
        tests = []

        # In-place rotations
        tests.append({
            'name': 'in_place_rotations',
            'instructions': ['L', 'L', 'R', 'R'],
            'description': 'Multiple in-place rotations'
        })

        # Moving rotations at low momentum
        tests.append({
            'name': 'moving_rotations_low_momentum',
            'instructions': ['F1', 'F1L', 'F1R'],
            'description': 'Moving rotations with m_eff <= 1'
        })

        # Moving rotations at high momentum (should crash)
        tests.append({
            'name': 'moving_rotations_high_momentum_crash',
            'instructions': ['F2', 'F2', 'F2L'],
            'description': 'Moving rotation with m_eff > 1 (crash expected)'
        })

        return tests

    def generate_corner_turn_tests(self) -> List[Dict[str, Any]]:
        """Generate test cases for corner turns."""
        tests = []

        # Tight turns at low momentum
        tests.append({
            'name': 'tight_turns_low_momentum',
            'instructions': ['F1', 'F1LT', 'F1RT'],
            'description': 'Tight corner turns with m_eff <= 1'
        })

        # Wide turns at higher momentum
        tests.append({
            'name': 'wide_turns_medium_momentum',
            'instructions': ['F2', 'F1', 'F1LW', 'F1RW'],
            'description': 'Wide corner turns with m_eff <= 2'
        })

        # Invalid corner turns (high momentum tight)
        tests.append({
            'name': 'invalid_tight_turn_high_momentum',
            'instructions': ['F2', 'F2', 'F2LT'],
            'description': 'Tight turn with m_eff > 1 (crash expected)'
        })

        return tests

    def generate_crash_tests(self) -> List[Dict[str, Any]]:
        """Generate test cases that should cause crashes."""
        tests = []

        # Direction change without stopping
        tests.append({
            'name': 'direction_change_without_stop',
            'instructions': ['F2', 'V1'],
            'description': 'Try to reverse without reaching 0 momentum (crash)'
        })

        # Invalid token
        tests.append({
            'name': 'invalid_token',
            'instructions': ['INVALID'],
            'description': 'Use unrecognized movement token (crash)'
        })

        # Rotation at non-zero momentum
        tests.append({
            'name': 'rotation_at_momentum',
            'instructions': ['F1', 'L'],
            'description': 'In-place rotation with momentum != 0 (crash)'
        })

        return tests

    def generate_goal_reaching_tests(self) -> List[Dict[str, Any]]:
        """Generate test sequences that might reach the goal."""
        tests = []

        # Simple path to goal (assuming open path)
        tests.append({
            'name': 'simple_path_to_goal',
            'instructions': ['F2'] * 7 + ['F1'] * 2 + ['BB'],  # Approximate path
            'description': 'Attempt to reach goal with forward movements'
        })

        # Path with turns
        tests.append({
            'name': 'path_with_turns',
            'instructions': ['F2'] * 3 + ['R'] + ['F2'] * 4 + ['L'] + ['F2'] * 3 + ['BB'],
            'description': 'Path with rotations to navigate'
        })

        return tests

    def generate_random_tests(self, num_tests: int = 10, max_length: int = 20) -> List[Dict[str, Any]]:
        """Generate random test sequences."""
        tests = []
        for i in range(num_tests):
            length = random.randint(1, max_length)
            instructions = [random.choice(ALL_TOKENS) for _ in range(length)]
            tests.append({
                'name': f'random_test_{i+1}',
                'instructions': instructions,
                'description': f'Random sequence of {length} commands'
            })
        return tests

    def generate_all_tests(self) -> List[Dict[str, Any]]:
        """Generate all test cases."""
        all_tests = []
        all_tests.extend(self.generate_basic_movement_tests())
        all_tests.extend(self.generate_rotation_tests())
        all_tests.extend(self.generate_corner_turn_tests())
        all_tests.extend(self.generate_crash_tests())
        all_tests.extend(self.generate_goal_reaching_tests())
        all_tests.extend(self.generate_random_tests())
        return all_tests

    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case against the API."""
        # Mock request data (would need to be updated based on actual state)
        request_data = {
            "game_uuid": "test-uuid",
            "sensor_data": [1, 1, 0, 1, 1],  # Mock sensor data
            "total_time_ms": 0,
            "goal_reached": False,
            "best_time_ms": None,
            "run_time_ms": 0,
            "run": 0,
            "momentum": 0
        }

        try:
            response = requests.post(
                f"{self.base_url}/micro-mouse",
                json=request_data,
                timeout=10
            )
            return {
                'test_name': test_case['name'],
                'instructions_sent': test_case['instructions'],
                'response': response.json() if response.status_code == 200 else None,
                'status_code': response.status_code,
                'success': response.status_code == 200
            }
        except Exception as e:
            return {
                'test_name': test_case['name'],
                'instructions_sent': test_case['instructions'],
                'error': str(e),
                'success': False
            }

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all generated test cases."""
        tests = self.generate_all_tests()
        results = []
        for test in tests:
            result = self.run_test(test)
            results.append(result)
            print(f"Test: {test['name']} - {'PASS' if result['success'] else 'FAIL'}")
        return results

    def save_tests_to_file(self, filename: str = "test_commands.json"):
        """Save generated test cases to a JSON file."""
        tests = self.generate_all_tests()
        with open(filename, 'w') as f:
            json.dump(tests, f, indent=2)
        print(f"Saved {len(tests)} test cases to {filename}")

if __name__ == "__main__":
    generator = CommandGenerator()

    # Generate and save test commands
    print("Generating test command sequences...")
    generator.save_tests_to_file()

    # Optionally run tests against the API
    print("\nRunning tests against API...")
    results = generator.run_all_tests()

    # Save results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Test results saved to test_results.json")
