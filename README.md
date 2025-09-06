# Micromouse Test Commands Generator

This codebase generates comprehensive test commands for testing the Micromouse API based on the provided requirements.

## Generated Test Commands

The `test_generate_commands.py` script creates `test_commands.json` with 24 different test scenarios covering:

### Basic Movement Tests
- **accelerate_from_rest**: F1, F2, F2, BB - Accelerate from 0 to 4, then brake
- **decelerate_sequence**: F2, F2, F1, F0, F0 - Accelerate then decelerate to 0
- **reverse_movements**: V1, V2, V2, BB - Reverse accelerate and brake

### Rotation Tests
- **in_place_rotations**: L, L, R, R - Multiple in-place rotations
- **moving_rotations_low_momentum**: F1, F1L, F1R - Moving rotations with m_eff ≤ 1
- **moving_rotations_high_momentum_crash**: F2, F2, F2L - Moving rotation with m_eff > 1 (crash expected)

### Corner Turn Tests
- **tight_turns_low_momentum**: F1, F1LT, F1RT - Tight corner turns with m_eff ≤ 1
- **wide_turns_medium_momentum**: F2, F1, F1LW, F1RW - Wide corner turns with m_eff ≤ 2
- **invalid_tight_turn_high_momentum**: F2, F2, F2LT - Tight turn with m_eff > 1 (crash expected)

### Crash Scenario Tests
- **direction_change_without_stop**: F2, V1 - Try to reverse without reaching 0 momentum (crash)
- **invalid_token**: INVALID - Use unrecognized movement token (crash)
- **rotation_at_momentum**: F1, L - In-place rotation with momentum ≠ 0 (crash)

### Goal Reaching Tests
- **simple_path_to_goal**: Multiple F2 commands - Attempt to reach goal with forward movements
- **path_with_turns**: F2 commands with rotations - Path with rotations to navigate

### Random Tests
- 10 randomly generated sequences of varying lengths

## Usage

### Generate Test Commands
```bash
python test_generate_commands.py
```
This creates `test_commands.json` with all test scenarios.

### Test Against Running API
1. Start the API server:
```bash
python main.py
```

2. Run all tests:
```bash
python test_commands.py --run-all
```

3. Run specific test:
```bash
python test_commands.py --test accelerate_from_rest
```

4. List available tests:
```bash
python test_commands.py --list
```

## Test Results
Results are saved to `command_test_results.json` with details about each test including:
- Success/failure status
- Instructions sent
- API response
- Any errors encountered

## Movement Token Coverage
The test suite covers all movement tokens from the requirements:
- Longitudinal: F0, F1, F2, BB, V0, V1, V2
- In-place rotations: L, R
- Moving rotations: F?L, F?R, V?L, V?R, BB?
- Corner turns: F?LT/LW/RT/RW, V?LT/LW/RT/RW

## Crash Testing
Specific tests are designed to trigger crash conditions:
- Momentum violations
- Invalid tokens
- Illegal direction changes
- Excessive momentum for turns

This ensures the API properly handles error conditions as specified in the requirements.
