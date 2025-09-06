# Micromouse Controller
# Constants and Data Structures

import heapq
import math
from typing import List, Tuple, Dict, Set, Optional
from enum import Enum

# Maze constants
MAZE_SIZE = 16
CELL_SIZE = 16  # cm
GOAL_CELLS = [(7, 7), (7, 8), (8, 7), (8, 8)]  # 2x2 center goal

# Directions (in degrees, 45° steps)
DIRECTIONS = {
    0: (0, 1),    # North
    45: (1, 1),   # Northeast
    90: (1, 0),   # East
    135: (1, -1), # Southeast
    180: (0, -1), # South
    225: (-1, -1),# Southwest
    270: (-1, 0), # West
    315: (-1, 1)  # Northwest
}

# Movement tokens
MOVEMENT_TOKENS = {
    'F0': 'Forward decel by 1',
    'F1': 'Forward hold',
    'F2': 'Forward accel by 1',
    'BB': 'Brake by 2',
    'V0': 'Reverse decel by 1',
    'V1': 'Reverse hold',
    'V2': 'Reverse accel by 1',
    'L': 'Rotate 45° CCW',
    'R': 'Rotate 45° CW',
    # Moving rotations
    'F0L': 'F0 + L', 'F0R': 'F0 + R',
    'F1L': 'F1 + L', 'F1R': 'F1 + R',
    'F2L': 'F2 + L', 'F2R': 'F2 + R',
    'BBL': 'BB + L', 'BBR': 'BB + R',
    'V0L': 'V0 + L', 'V0R': 'V0 + R',
    'V1L': 'V1 + L', 'V1R': 'V1 + R',
    'V2L': 'V2 + L', 'V2R': 'V2 + R',
    # Corner turns
    'F0LT': 'F0 + Left Tight', 'F0LW': 'F0 + Left Wide',
    'F0RT': 'F0 + Right Tight', 'F0RW': 'F0 + Right Wide',
    'F1LT': 'F1 + Left Tight', 'F1LW': 'F1 + Left Wide',
    'F1RT': 'F1 + Right Tight', 'F1RW': 'F1 + Right Wide',
    'F2LT': 'F2 + Left Tight', 'F2LW': 'F2 + Left Wide',
    'F2RT': 'F2 + Right Tight', 'F2RW': 'F2 + Right Wide',
    'V0LT': 'V0 + Left Tight', 'V0LW': 'V0 + Left Wide',
    'V0RT': 'V0 + Right Tight', 'V0RW': 'V0 + Right Wide',
    'V1LT': 'V1 + Left Tight', 'V1LW': 'V1 + Left Wide',
    'V1RT': 'V1 + Right Tight', 'V1RW': 'V1 + Right Wide',
    'V2LT': 'V2 + Left Tight', 'V2LW': 'V2 + Left Wide',
    'V2RT': 'V2 + Right Tight', 'V2RW': 'V2 + Right Wide',
    # Corner turns with end rotation
    'F0LTL': 'F0 + Left Tight + L', 'F0LTR': 'F0 + Left Tight + R',
    'F0LWL': 'F0 + Left Wide + L', 'F0LWR': 'F0 + Left Wide + R',
    'F0RTL': 'F0 + Right Tight + L', 'F0RTR': 'F0 + Right Tight + R',
    'F0RWL': 'F0 + Right Wide + L', 'F0RWR': 'F0 + Right Wide + R',
    'F1LTL': 'F1 + Left Tight + L', 'F1LTR': 'F1 + Left Tight + R',
    'F1LWL': 'F1 + Left Wide + L', 'F1LWR': 'F1 + Left Wide + R',
    'F1RTL': 'F1 + Right Tight + L', 'F1RTR': 'F1 + Right Tight + R',
    'F1RWL': 'F1 + Right Wide + L', 'F1RWR': 'F1 + Right Wide + R',
    'F2LTL': 'F2 + Left Tight + L', 'F2LTR': 'F2 + Left Tight + R',
    'F2LWL': 'F2 + Left Wide + L', 'F2LWR': 'F2 + Left Wide + R',
    'F2RTL': 'F2 + Right Tight + L', 'F2RTR': 'F2 + Right Tight + R',
    'F2RWL': 'F2 + Right Wide + L', 'F2RWR': 'F2 + Right Wide + R',
    'V0LTL': 'V0 + Left Tight + L', 'V0LTR': 'V0 + Left Tight + R',
    'V0LWL': 'V0 + Left Wide + L', 'V0LWR': 'V0 + Left Wide + R',
    'V0RTL': 'V0 + Right Tight + L', 'V0RTR': 'V0 + Right Tight + R',
    'V0RWL': 'V0 + Right Wide + L', 'V0RWR': 'V0 + Right Wide + R',
    'V1LTL': 'V1 + Left Tight + L', 'V1LTR': 'V1 + Left Tight + R',
    'V1LWL': 'V1 + Left Wide + L', 'V1LWR': 'V1 + Left Wide + R',
    'V1RTL': 'V1 + Right Tight + L', 'V1RTR': 'V1 + Right Tight + R',
    'V1RWL': 'V1 + Right Wide + L', 'V1RWR': 'V1 + Right Wide + R',
    'V2LTL': 'V2 + Left Tight + L', 'V2LTR': 'V2 + Left Tight + R',
    'V2LWL': 'V2 + Left Wide + L', 'V2LWR': 'V2 + Left Wide + R',
    'V2RTL': 'V2 + Right Tight + L', 'V2RTR': 'V2 + Right Tight + R',
    'V2RWL': 'V2 + Right Wide + L', 'V2RWR': 'V2 + Right Wide + R',
}

# Base times (ms)
BASE_TIMES = {
    'half_step': 500,
    'diagonal_half_step': 600,
    'tight_corner': 700,
    'wide_corner': 1400,
    'in_place_turn': 200,
    'default_rest': 200,
}

# Momentum reduction table
MOMENTUM_REDUCTION = {
    0.0: 0.00,
    0.5: 0.10,
    1.0: 0.20,
    1.5: 0.275,
    2.0: 0.35,
    2.5: 0.40,
    3.0: 0.45,
    3.5: 0.475,
    4.0: 0.50,
}

# Sensor angles (degrees)
SENSOR_ANGLES = [-90, -45, 0, 45, 90]
SENSOR_RANGE = 12  # cm

# Grid cell states
class CellState(Enum):
    UNKNOWN = 0
    OPEN = 1
    WALL = 2
    GOAL = 3

# Orientation steps (45° increments)
ORIENTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]

def get_direction_vector(orientation: int) -> Tuple[int, int]:
    """Get the (dx, dy) vector for a given orientation."""
    # Normalize to 0-315 degrees in 45° steps
    normalized = (orientation % 360) // 45 * 45
    return DIRECTIONS.get(normalized, (0, 0))

def get_momentum_reduction(m_eff: float) -> float:
    """Get time reduction based on effective momentum."""
    m_eff = min(max(m_eff, 0.0), 4.0)
    # Linear interpolation between points
    keys = sorted(MOMENTUM_REDUCTION.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= m_eff <= keys[i+1]:
            frac = (m_eff - keys[i]) / (keys[i+1] - keys[i])
            return MOMENTUM_REDUCTION[keys[i]] + frac * (MOMENTUM_REDUCTION[keys[i+1]] - MOMENTUM_REDUCTION[keys[i]])
    return MOMENTUM_REDUCTION[keys[-1]]

def calculate_m_eff(m_in: int, m_out: int) -> float:
    """Calculate effective momentum."""
    return (abs(m_in) + abs(m_out)) / 2.0

def is_cardinal(orientation: int) -> bool:
    """Check if orientation is cardinal (0, 90, 180, 270)."""
    return orientation % 90 == 0

class State:
    """Represents the current state of the micromouse."""
    
    def __init__(self):
        self.position: Tuple[float, float] = (0.5, 0.5)  # Start at center of (0,0) cell
        self.orientation: int = 0  # Facing North
        self.momentum: int = 0
        self.grid: List[List[CellState]] = [[CellState.UNKNOWN for _ in range(MAZE_SIZE)] for _ in range(MAZE_SIZE)]
        self.walls: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()  # Wall between cells
        self.visited: Set[Tuple[int, int]] = set()
        
        # Mark goal cells
        for gx, gy in GOAL_CELLS:
            self.grid[gx][gy] = CellState.GOAL
    
    def get_cell(self) -> Tuple[int, int]:
        """Get the current cell coordinates."""
        return (int(self.position[0]), int(self.position[1]))
    
    def update_position(self, dx: float, dy: float):
        """Update position by delta."""
        self.position = (self.position[0] + dx, self.position[1] + dy)
    
    def set_orientation(self, new_orientation: int):
        """Set orientation, normalized to 0-315."""
        self.orientation = new_orientation % 360
    
    def is_in_goal(self) -> bool:
        """Check if mouse is in goal interior."""
        x, y = self.position
        # Goal is interior of 2x2 center cells 7,7 7,8 8,7 8,8
        return 7.0 <= x < 9.0 and 7.0 <= y < 9.0
    
    def mark_visited(self):
        """Mark current cell as visited."""
        cell = self.get_cell()
        self.visited.add(cell)
        if self.grid[cell[0]][cell[1]] == CellState.UNKNOWN:
            self.grid[cell[0]][cell[1]] = CellState.OPEN
    
    def add_wall(self, cell1: Tuple[int, int], cell2: Tuple[int, int]):
        """Add a wall between two cells."""
        wall = (min(cell1, cell2), max(cell1, cell2))
        self.walls.add(wall)
        # Mark cells as walls if necessary
        for cell in [cell1, cell2]:
            if 0 <= cell[0] < MAZE_SIZE and 0 <= cell[1] < MAZE_SIZE:
                if self.grid[cell[0]][cell[1]] == CellState.UNKNOWN:
                    self.grid[cell[0]][cell[1]] = CellState.WALL

class Mapper:
    """Handles maze mapping and frontier exploration."""
    
    def __init__(self, state: State):
        self.state = state
    
    def update_from_sensors(self, sensor_data: List[int]):
        """Update grid based on sensor readings."""
        current_cell = self.state.get_cell()
        orientation = self.state.orientation
        
        for i, distance in enumerate(sensor_data):
            angle = SENSOR_ANGLES[i]
            absolute_angle = (orientation + angle) % 360
            
            # Calculate hit point
            rad_angle = math.radians(absolute_angle)
            hit_x = self.state.position[0] + distance * math.cos(rad_angle)
            hit_y = self.state.position[1] + distance * math.sin(rad_angle)
            
            hit_cell = (int(hit_x), int(hit_y))
            
            # If sensor hit a wall within range, mark wall
            if distance < SENSOR_RANGE:
                # Wall at hit_cell
                if 0 <= hit_cell[0] < MAZE_SIZE and 0 <= hit_cell[1] < MAZE_SIZE:
                    self.state.grid[hit_cell[0]][hit_cell[1]] = CellState.WALL
                # Add wall between current cell and adjacent
                adjacent = self.get_adjacent_cell(current_cell, absolute_angle)
                if adjacent:
                    self.state.add_wall(current_cell, adjacent)
            else:
                # Clear path, mark cells as open
                steps = int(distance / CELL_SIZE)
                for step in range(1, steps + 1):
                    check_x = self.state.position[0] + step * CELL_SIZE * math.cos(rad_angle)
                    check_y = self.state.position[1] + step * CELL_SIZE * math.sin(rad_angle)
                    check_cell = (int(check_x), int(check_y))
                    if 0 <= check_cell[0] < MAZE_SIZE and 0 <= check_cell[1] < MAZE_SIZE:
                        if self.state.grid[check_cell[0]][check_cell[1]] == CellState.UNKNOWN:
                            self.state.grid[check_cell[0]][check_cell[1]] = CellState.OPEN
    
    def get_adjacent_cell(self, cell: Tuple[int, int], angle: int) -> Optional[Tuple[int, int]]:
        """Get adjacent cell in the direction of the angle."""
        dx, dy = get_direction_vector(angle)
        adjacent = (cell[0] + dx, cell[1] + dy)
        if 0 <= adjacent[0] < MAZE_SIZE and 0 <= adjacent[1] < MAZE_SIZE:
            return adjacent
        return None
    
    def get_frontier(self) -> List[Tuple[int, int]]:
        """Get unexplored frontier cells."""
        frontier = []
        for x in range(MAZE_SIZE):
            for y in range(MAZE_SIZE):
                if self.state.grid[x][y] == CellState.UNKNOWN:
                    # Check if adjacent to known open cell
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < MAZE_SIZE and 0 <= ny < MAZE_SIZE:
                            if self.state.grid[nx][ny] == CellState.OPEN:
                                frontier.append((x, y))
                                break
        return frontier
    
    def is_mapped(self) -> bool:
        """Check if the maze is fully mapped."""
        return all(cell != CellState.UNKNOWN for row in self.state.grid for cell in row)

class Planner:
    """Handles path planning using A* algorithm."""
    
    def __init__(self, state: State):
        self.state = state
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cell[0] + dx, cell[1] + dy
            if 0 <= nx < MAZE_SIZE and 0 <= ny < MAZE_SIZE:
                # Check if there's a wall between cells
                wall = (min(cell, (nx, ny)), max(cell, (nx, ny)))
                if wall not in self.state.walls and self.state.grid[nx][ny] != CellState.WALL:
                    neighbors.append((nx, ny))
        return neighbors
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding from start to goal."""
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            
            if current == goal:
                break
            
            for neighbor in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1  # Each move costs 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
        
        if goal not in came_from:
            return None
        
        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
    
    def find_path_to_frontier(self, start: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find path to nearest frontier cell."""
        mapper = Mapper(self.state)
        frontiers = mapper.get_frontier()
        if not frontiers:
            return None
        
        min_path = None
        min_dist = float('inf')
        
        for frontier in frontiers:
            path = self.a_star(start, frontier)
            if path and len(path) < min_dist:
                min_dist = len(path)
                min_path = path
        
        return min_path
    
    def find_path_to_goal(self, start: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find shortest path to goal area."""
        # Find path to nearest goal cell
        min_path = None
        min_dist = float('inf')
        
        for goal_cell in GOAL_CELLS:
            path = self.a_star(start, goal_cell)
            if path and len(path) < min_dist:
                min_dist = len(path)
                min_path = path
        
        return min_path

class SpeedProfiler:
    """Generates optimal movement token sequences with momentum management."""
    
    def __init__(self, state: State):
        self.state = state
    
    def path_to_tokens(self, path: List[Tuple[int, int]]) -> List[str]:
        """Convert path to movement tokens."""
        if not path or len(path) < 2:
            return []
        
        tokens = []
        current_pos = path[0]
        current_orientation = self.state.orientation
        current_momentum = self.state.momentum
        
        for i in range(1, len(path)):
            next_pos = path[i]
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            
            # Determine required orientation
            if dx == 1 and dy == 0:
                required_orientation = 90   # East
            elif dx == -1 and dy == 0:
                required_orientation = 270  # West
            elif dx == 0 and dy == 1:
                required_orientation = 0    # North
            elif dx == 0 and dy == -1:
                required_orientation = 180  # South
            else:
                continue  # Invalid move
            
            # Calculate turn needed
            turn_angle = (required_orientation - current_orientation) % 360
            if turn_angle > 180:
                turn_angle -= 360
            
            # Add turn tokens if needed
            if turn_angle != 0:
                turn_tokens = self.generate_turn_tokens(turn_angle, current_momentum)
                tokens.extend(turn_tokens)
                current_orientation = required_orientation
            
            # Add forward move
            move_token = self.generate_forward_token(current_momentum)
            tokens.append(move_token)
            current_momentum = self.update_momentum(current_momentum, move_token)
            
            current_pos = next_pos
        
        # Final stop in goal
        stop_tokens = self.generate_stop_tokens(current_momentum)
        tokens.extend(stop_tokens)
        
        return tokens
    
    def generate_turn_tokens(self, turn_angle: int, momentum: int) -> List[str]:
        """Generate tokens for turning."""
        tokens = []
        steps = abs(turn_angle) // 45
        
        if momentum != 0:
            # Must stop first for in-place turns
            tokens.extend(self.generate_stop_tokens(momentum))
            momentum = 0
        
        turn_token = 'L' if turn_angle > 0 else 'R'
        tokens.extend([turn_token] * steps)
        
        return tokens
    
    def generate_forward_token(self, momentum: int) -> str:
        """Generate forward movement token based on current momentum."""
        if momentum < 4:
            return 'F2'  # Accelerate
        else:
            return 'F1'  # Cruise
    
    def generate_stop_tokens(self, momentum: int) -> List[str]:
        """Generate tokens to stop from current momentum."""
        tokens = []
        current_m = momentum
        
        while current_m != 0:
            if abs(current_m) <= 2:
                tokens.append('F0' if current_m > 0 else 'V0')
                current_m += -1 if current_m > 0 else 1
            else:
                tokens.append('BB')
                current_m += -2 if current_m > 0 else 2
        
        return tokens
    
    def update_momentum(self, momentum: int, token: str) -> int:
        """Update momentum based on token."""
        if token == 'F2':
            return min(momentum + 1, 4)
        elif token == 'F1':
            return momentum
        elif token == 'F0':
            return max(momentum - 1, 0) if momentum > 0 else min(momentum + 1, 0)
        elif token == 'BB':
            if momentum > 0:
                return max(momentum - 2, 0)
            elif momentum < 0:
                return min(momentum + 2, 0)
        return momentum

class SafetyGuard:
    """Ensures moves are safe before execution."""
    
    def __init__(self, state: State):
        self.state = state
    
    def is_safe_batch(self, tokens: List[str], sensor_data: List[int]) -> bool:
        """Check if a batch of tokens is safe to execute."""
        # Simplified: check if front sensor sees enough clearance
        if len(sensor_data) >= 3:
            front_sensor = sensor_data[2]  # 0° sensor
            if front_sensor < 16:  # Need at least 16 cm for safe move
                return False
        
        # Check momentum constraints for each token
        simulated_momentum = self.state.momentum
        simulated_orientation = self.state.orientation
        
        for token in tokens:
            if not self.is_token_safe(token, simulated_momentum, simulated_orientation):
                return False
            simulated_momentum = self.simulate_momentum_update(simulated_momentum, token)
            if 'L' in token or 'R' in token:
                simulated_orientation = self.simulate_orientation_update(simulated_orientation, token)
        
        return True
    
    def is_token_safe(self, token: str, momentum: int, orientation: int) -> bool:
        """Check if a single token is safe."""
        # Check momentum sign consistency
        if 'F' in token and momentum < 0:
            return False
        if 'V' in token and momentum > 0:
            return False
        
        # Check in-place turn constraints
        if token in ['L', 'R'] and momentum != 0:
            return False
        
        # Check moving rotation constraints
        if any(t in token for t in ['F', 'V', 'BB']) and ('L' in token or 'R' in token):
            m_eff = calculate_m_eff(momentum, self.simulate_momentum_update(momentum, token.split('L')[0].split('R')[0]))
            if m_eff > 1:
                return False
        
        # Check corner turn constraints
        if 'T' in token or 'W' in token:
            if not is_cardinal(orientation):
                return False
            m_eff = calculate_m_eff(momentum, self.simulate_momentum_update(momentum, token[:2]))
            if 'T' in token and m_eff > 1:
                return False
            if 'W' in token and m_eff > 2:
                return False
        
        return True
    
    def simulate_momentum_update(self, momentum: int, token: str) -> int:
        """Simulate momentum change for a token."""
        base_token = token[:2] if len(token) > 2 else token
        if base_token == 'F2':
            return min(momentum + 1, 4)
        elif base_token == 'F1':
            return momentum
        elif base_token == 'F0':
            return max(momentum - 1, 0) if momentum > 0 else min(momentum + 1, 0)
        elif base_token == 'BB':
            if momentum > 0:
                return max(momentum - 2, 0)
            elif momentum < 0:
                return min(momentum + 2, 0)
        elif base_token == 'V2':
            return max(momentum - 1, -4)
        elif base_token == 'V1':
            return momentum
        elif base_token == 'V0':
            return min(momentum + 1, 0) if momentum < 0 else max(momentum - 1, 0)
        return momentum
    
    def simulate_orientation_update(self, orientation: int, token: str) -> int:
        """Simulate orientation change for a token."""
        if 'T' in token or 'W' in token:
            # Corner turn - extract turn direction
            if 'L' in token and 'T' in token:
                turn_dir = 'L'
            elif 'R' in token and 'T' in token:
                turn_dir = 'R'
            elif 'L' in token and 'W' in token:
                turn_dir = 'L'
            elif 'R' in token and 'W' in token:
                turn_dir = 'R'
            else:
                turn_dir = None
            
            if turn_dir == 'L':
                orientation = (orientation - 90) % 360
            elif turn_dir == 'R':
                orientation = (orientation + 90) % 360
            
            # Check for additional rotation at end
            if token.endswith('L'):
                orientation = (orientation - 45) % 360
            elif token.endswith('R'):
                orientation = (orientation + 45) % 360
        elif 'L' in token and not ('T' in token or 'W' in token):
            orientation = (orientation - 45) % 360
        elif 'R' in token and not ('T' in token or 'W' in token):
            orientation = (orientation + 45) % 360
        
        return orientation

class MainController:
    """Main controller for the micromouse."""
    
    def __init__(self):
        self.state = State()
        self.mapper = Mapper(self.state)
        self.planner = Planner(self.state)
        self.speed_profiler = SpeedProfiler(self.state)
        self.safety_guard = SafetyGuard(self.state)
        
        self.phase = 'mapping'  # 'mapping' or 'speedrun'
        self.final_path = None
        
        # Game state tracking
        self.game_uuid = None
        self.total_time_ms = 0
        self.run_time_ms = 0
        self.run = 0
        self.goal_reached = False
        self.best_time_ms = None
        self.challenge_ended = False
    
    def process_request(self, request_data: Dict) -> Dict:
        """Process API request and return response."""
        # Update game state from request
        self.game_uuid = request_data.get('game_uuid')
        sensor_data = request_data.get('sensor_data', [12, 12, 12, 12, 12])
        instructions = request_data.get('instructions', [])
        
        # Update timing and state
        old_total_time = self.total_time_ms
        old_run_time = self.run_time_ms
        old_goal_reached = self.goal_reached
        
        self.total_time_ms = request_data.get('total_time_ms', 0)
        self.run_time_ms = request_data.get('run_time_ms', 0)
        self.run = request_data.get('run', 0)
        self.goal_reached = request_data.get('goal_reached', False)
        self.best_time_ms = request_data.get('best_time_ms')
        
        # Update mouse state
        self.state.momentum = request_data.get('momentum', 0)
        
        # Check for end flag
        if request_data.get('end', False):
            return {'instructions': [], 'end': False}  # Don't process instructions
        
        # Process instructions if provided
        if instructions:
            # Add thinking time (50ms per batch)
            self.total_time_ms += 50
            
            # Execute instructions and calculate time
            execution_time = self.execute_instructions(instructions, sensor_data)
            self.total_time_ms += execution_time
            self.run_time_ms += execution_time
            
            # Check if goal reached
            if self.state.is_in_goal() and self.state.momentum == 0:
                if not old_goal_reached:
                    self.goal_reached = True
                    if self.best_time_ms is None or self.run_time_ms < self.best_time_ms:
                        self.best_time_ms = self.run_time_ms
        
        # Update mapping from sensors
        # self.mapper.update_from_sensors(sensor_data)
        self.state.mark_visited()
        
        # Generate response instructions (for autonomous mode)
        response_instructions = []
        
        if not instructions:  # Only generate if no instructions were sent
            if self.phase == 'mapping' and self.mapper.is_mapped():
                self.phase = 'speedrun'
                self.final_path = self.planner.find_path_to_goal(self.state.get_cell())
            
            if self.phase == 'mapping':
                path = self.planner.find_path_to_frontier(self.state.get_cell())
                if path:
                    response_instructions = self.speed_profiler.path_to_tokens(path[:3])
            elif self.phase == 'speedrun' and self.final_path:
                response_instructions = self.speed_profiler.path_to_tokens(self.final_path)
        
        return {
            'instructions': response_instructions,
            'end': False
        }
    
    def execute_instructions(self, instructions: List[str], sensor_data: List[int]) -> int:
        """Execute movement instructions and return total execution time."""
        total_time = 0
        crash = False
        
        for token in instructions:
            if crash:
                break
                
            # Validate token
            if not self.is_valid_token(token):
                crash = True
                break
            
            # Check safety constraints
            if not self.safety_guard.is_token_safe(token, self.state.momentum, self.state.orientation):
                crash = True
                break
            
            # Execute movement
            time_taken = self.execute_token(token)
            total_time += time_taken
            
            # Update position based on movement
            self.update_position_from_token(token)
        
        if crash:
            self.challenge_ended = True
            return total_time  # Return time up to crash
        
        return total_time
    
    def is_valid_token(self, token: str) -> bool:
        """Check if token is valid according to requirements."""
        return token in MOVEMENT_TOKENS
    
    def execute_token(self, token: str) -> int:
        """Execute a movement token and return time taken."""
        base_time = 0
        m_in = self.state.momentum
        
        # Determine base time and momentum change
        if token in ['F0', 'F1', 'F2', 'V0', 'V1', 'V2', 'BB']:
            # Longitudinal movement
            if token == 'BB':
                base_time = BASE_TIMES['default_rest'] if m_in == 0 else BASE_TIMES['half_step']
            else:
                # Check if movement is diagonal (intercardinal)
                orientation = self.state.orientation
                is_diagonal = orientation % 90 != 0
                base_time = BASE_TIMES['diagonal_half_step'] if is_diagonal else BASE_TIMES['half_step']
        elif token in ['L', 'R']:
            # In-place rotation
            base_time = BASE_TIMES['in_place_turn']
        elif 'L' in token or 'R' in token:
            # Moving rotation or corner turn
            if 'T' in token or 'W' in token:
                # Corner turn
                base_time = BASE_TIMES['tight_corner'] if 'T' in token else BASE_TIMES['wide_corner']
            else:
                # Moving rotation
                base_time = BASE_TIMES['half_step']
        
        # Calculate momentum out
        m_out = self.calculate_momentum_out(token)
        
        # Apply momentum reduction
        m_eff = calculate_m_eff(m_in, m_out)
        reduction = get_momentum_reduction(m_eff)
        actual_time = int(base_time * (1 - reduction))
        
        # Update momentum
        self.state.momentum = m_out
        
        # Update orientation
        if 'T' in token or 'W' in token:
            # Corner turn - extract turn direction
            if 'L' in token and 'T' in token:
                turn_dir = 'L'
            elif 'R' in token and 'T' in token:
                turn_dir = 'R'
            elif 'L' in token and 'W' in token:
                turn_dir = 'L'
            elif 'R' in token and 'W' in token:
                turn_dir = 'R'
            else:
                turn_dir = None
            
            if turn_dir == 'L':
                self.state.orientation = (self.state.orientation - 90) % 360
            elif turn_dir == 'R':
                self.state.orientation = (self.state.orientation + 90) % 360
            
            # Check for additional rotation at end
            if token.endswith('L'):
                self.state.orientation = (self.state.orientation - 45) % 360
            elif token.endswith('R'):
                self.state.orientation = (self.state.orientation + 45) % 360
        elif 'L' in token and not ('T' in token or 'W' in token):
            self.state.orientation = (self.state.orientation - 45) % 360
        elif 'R' in token and not ('T' in token or 'W' in token):
            self.state.orientation = (self.state.orientation + 45) % 360
        
        return actual_time
    
    def calculate_momentum_out(self, token: str) -> int:
        """Calculate momentum after executing token."""
        m = self.state.momentum
        
        if token == 'F2':
            return min(m + 1, 4)
        elif token == 'F1':
            return m
        elif token == 'F0':
            return max(m - 1, 0) if m > 0 else m
        elif token == 'V2':
            return max(m - 1, -4)
        elif token == 'V1':
            return m
        elif token == 'V0':
            return min(m + 1, 0) if m < 0 else m
        elif token == 'BB':
            if m > 0:
                return max(m - 2, 0)
            elif m < 0:
                return min(m + 2, 0)
            else:
                return 0
        
        # For complex tokens, extract base movement
        base_token = token[:2] if len(token) >= 2 else token
        return self.calculate_momentum_out(base_token)
    
    def update_position_from_token(self, token: str):
        """Update position based on movement token."""
        # Simplified position update - in real implementation would be more complex
        dx, dy = 0, 0
        
        if token in ['F0', 'F1', 'F2', 'BB'] and self.state.momentum > 0:
            # Forward movement
            rad_angle = math.radians(self.state.orientation)
            is_diagonal = self.state.orientation % 90 != 0
            distance = 11 if is_diagonal else 8  # ~11cm for diagonal, 8cm for cardinal
            dx = distance * math.cos(rad_angle)
            dy = distance * math.sin(rad_angle)
        elif token in ['V0', 'V1', 'V2', 'BB'] and self.state.momentum < 0:
            # Reverse movement
            rad_angle = math.radians(self.state.orientation)
            is_diagonal = self.state.orientation % 90 != 0
            distance = 11 if is_diagonal else 8  # ~11cm for diagonal, 8cm for cardinal
            dx = -distance * math.cos(rad_angle)
            dy = -distance * math.sin(rad_angle)
        
        self.state.update_position(dx, dy)

# FastAPI setup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()
controller = MainController()

class MicroMouseRequest(BaseModel):
    game_uuid: str
    sensor_data: List[int]
    total_time_ms: int
    goal_reached: bool
    best_time_ms: Optional[int]
    run_time_ms: int
    run: int
    momentum: int
    instructions: List[str] = []  # Client sends instructions to execute
    end: bool = False  # Client can end the challenge

class MicroMouseResponse(BaseModel):
    instructions: List[str]
    end: bool

@app.post("/micro-mouse", response_model=MicroMouseResponse)
async def micro_mouse_endpoint(request: MicroMouseRequest):
    """Handle micromouse API requests."""
    try:
        response = controller.process_request(request.dict())
        return MicroMouseResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)