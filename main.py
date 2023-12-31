import heapq

class QStarAgent:
    def __init__(self, learning_rate, discount_factor, heuristic_function, possible_actions_function):
        self.q_table = {}  # Initialize Q-table
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.heuristic_function = heuristic_function  # Heuristic for A*
        self.possible_actions = possible_actions_function  # Function to get possible actions

    def learn(self, state, action, reward, next_state):
        # Standard Q-learning update
        old_value = self.q_table.get((state, action), 0)
        next_max = max(self.q_table.get((next_state, a), 0) for a in self.possible_actions(next_state))
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        print(f"Old value: {old_value}, Next max: {next_max}, New value: {new_value}")
        self.q_table[(state, action)] = new_value
    
    def get_neighbors(self, state):
        x, y = state
        potential_neighbors = [
            (x - 1, y),  # left
            (x + 1, y),  # right
            (x, y - 1),  # up
            (x, y + 1)   # down
        ]

        # Filter out invalid neighbors
        valid_neighbors = [neighbor for neighbor in potential_neighbors if self.is_valid_state(neighbor)]
        return valid_neighbors
    
    def cost(self, current_state, next_state):
        # Define the cost of moving from the current state to the next state
        # For a simple implementation, you can use a constant cost for each move
        return 1  # Example: a constant cost of 1 for each move



    def a_star_search(self, start_state, goal_state):
        # Priority queue to hold states to explore, format: (priority, state)
        frontier = []
        heapq.heappush(frontier, (0, start_state))

        # Stores the cost to reach each state from start
        cost_so_far = {start_state: 0}

        # Stores the parent of each state
        came_from = {start_state: None}

        while frontier:
            _, current_state = heapq.heappop(frontier)
            print(f"Current state: {current_state}, Frontier: {frontier}")
            # Check if goal is reached
            if current_state == goal_state:
                break

            for next_state in self.get_neighbors(current_state):
                # Update the cost for the next state
                new_cost = cost_so_far[current_state] + self.cost(current_state, next_state)
                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    priority = new_cost + self.heuristic(next_state, goal_state)
                    print(f"New cost: {new_cost}, Priority: {priority}, Next state: {next_state}")
                    heapq.heappush(frontier, (priority, next_state))
                    came_from[next_state] = current_state

        # Reconstruct path from start to goal
        path = []
        while current_state:
            path.append(current_state)
            current_state = came_from[current_state]
        path.reverse()
        return path

    def heuristic(self, state, goal_state):
        # Heuristic function that uses Q-values, you can modify this based on your application
        # Example: negative of the max Q-value for the given state, guiding towards higher Q-values
        return -max(self.q_table.get((state, action), 0) for action in self.possible_actions(state))


    def choose_action(self, state, goal_state, environment):
        # First, get all possible actions for the current state
        possible_actions = self.possible_actions(state)
        print(f"Current state: {state}, Possible actions: {possible_actions}")
        # If there are no possible actions, return None or a default action
        if not possible_actions:
            return None

        # Use A* to find the most promising next state towards the goal
        best_path = self.a_star_search(state, goal_state)
        if len(best_path) < 2:  # If A* doesn't provide a next step, use default Q-learning
            return max(possible_actions, key=lambda a: self.q_table.get((state, a), 0))

        next_state_on_path = best_path[1]  # Next state in the A* path

        # Choose the action that leads to the next_state_on_path
        # This requires knowing how actions translate to next states, which depends on your environment
        best_action = None
        best_action_value = float('-inf')
        for action in possible_actions:
            predicted_next_state = self.predict_next_state(state, action)
            if predicted_next_state == next_state_on_path:
                action_value = self.q_table.get((state, action), 0)
                if action_value > best_action_value:
                    best_action = action
                    best_action_value = action_value
        print(f"Chosen action: {best_action}")
        return best_action

    def predict_next_state(self, current_state, action):
        # The implementation here is environment-specific. 
        # You need to define how actions change the state.
        # For example, in a grid environment, actions might be movements in different directions.

        # Example for a grid environment:
        next_state = list(current_state)  # Assuming current_state is a tuple/list (x, y)
        print(f"Current state: {current_state}, Action: {action}, Next state: {next_state}")
        if action == 'up':
            next_state[1] -= 1  # Move up in the grid
        elif action == 'down':
            next_state[1] += 1  # Move down in the grid
        elif action == 'left':
            next_state[0] -= 1  # Move left in the grid
        elif action == 'right':
            next_state[0] += 1  # Move right in the grid

        # Make sure the new state is valid (within the bounds of the environment, not hitting obstacles, etc.)
        if self.is_valid_state(next_state):
            return tuple(next_state)
        else:
            return current_state  # Return the original state if the move is invalid

    def is_valid_state(self, state):
        # Example for a grid environment:
        # Assuming the environment has boundaries and potentially obstacles

        # Parameters for your grid environment
        width = 10  # Example width
        height = 10  # Example height
        start = (0, 0)  # Starting position
        goal = (9, 9)  # Goal position
        obstacles = [(5, 5), (6, 6)]  # List of obstacle positions

        # Check if the state is within the grid boundaries
        x, y = state
        # if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
        if x < 0 or x >= width or y < 0 or y >= height:
            return False  # State is outside grid boundaries

        # Check if the state is not an obstacle
        if state in obstacles:
            return False  # State is an obstacle

        # Additional checks can be added here, depending on the environment's rules

        return True  # State is valid


class GridEnvironment:
    def __init__(self, width, height, start, goal, obstacles):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles)
        self.current_state = start

    def reset(self):
        self.current_state = self.start
        return self.current_state

    def step(self, action):
        # Define the result of an action
        next_state = self.predict_next_state(self.current_state, action)
        reward = self.get_reward(next_state)
        done = self.is_done(next_state)
        self.current_state = next_state
        return next_state, reward, done

    def predict_next_state(self, current_state, action):
        # Assuming current_state is a tuple (x, y) representing a position in the grid
        x, y = current_state
    
        # Define how actions change the state
        if action == 'up':
            y -= 1  # Move up in the grid
        elif action == 'down':
            y += 1  # Move down in the grid
        elif action == 'left':
            x -= 1  # Move left in the grid
        elif action == 'right':
            x += 1  # Move right in the grid
    
        # Create the potential next state
        next_state = (x, y)
    
        # Check if the next state is valid
        if self.is_valid_state(next_state):
            return next_state
        else:
            return current_state  # If the move is invalid, return the original state


    def get_reward(self, state):
        # Define the reward system
        if state == self.goal:
            return 100  # High reward for reaching the goal
        elif state in self.obstacles:
            return -100  # High penalty for hitting an obstacle
        else:
            return -1  # Slight penalty to encourage shortest path

    def is_done(self, state):
        # Define when an episode is done
        return state == self.goal or state in self.obstacles

    def get_goal_state(self):
        return self.goal

    def get_current_state(self):
        return self.current_state

    def possible_actions(self, state):
        # Define the possible actions from a given state
        return ['up', 'down', 'left', 'right']

    def is_valid_state(self, state):
        x, y = state
    
        # Check if the state is within the grid boundaries
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False  # State is outside grid boundaries
    
        # Check if the state is an obstacle
        if state in self.obstacles:
            return False  # State is an obstacle
    
        return True  # State is valid if it's inside the grid and not an obstacle

# set up
total_episodes=1000

# Parameters for your grid environment
width = 10  # Example width
height = 10  # Example height
start = (0, 0)  # Starting position
goal = (9, 9)  # Goal position
obstacles = [(5, 5), (6, 6)]  # List of obstacle positions

# Create an instance of the GridEnvironment
environment = GridEnvironment(width, height, start, goal, obstacles)
# Define the heuristic function for A*
def your_heuristic(state, goal_state):
    # This is a placeholder heuristic function.
    # You need to define it based on your specific environment.
    # For example, in a grid environment, it could be the Manhattan distance to the goal state.
    return abs(state[0] - goal_state[0]) + abs(state[1] - goal_state[1])

# Instantiate the QStarAgent
agent = QStarAgent(learning_rate=0.1, discount_factor=0.9, heuristic_function=your_heuristic, possible_actions_function=environment.possible_actions)


# Training process
for episode in range(total_episodes):
    state = environment.reset()  # Reset the environment to the starting state
    goal_state = environment.get_goal_state()  # Define the goal state for this episode

    while not environment.is_done(state):
        # Select an action
        action = agent.choose_action(state, goal_state, environment)

        # Perform the action in the environment
        next_state, reward, done = environment.step(action)
        print(f"State: {state}, Action: {action}, Next state: {next_state}, Reward: {reward}")
        # Learn from the action's results
        agent.learn(state, action, reward, next_state)

        # Move to the next state
        state = next_state

    # Additional episode-wise logic (like updating exploration rates, logging, etc.) goes here

# Decision making with the trained agent
current_state = environment.get_current_state()
goal_state = environment.get_goal_state()

while not environment.is_done(state):
    # Choose the best action based on the trained model
    action = agent.choose_action(current_state, goal_state, environment)
    
    # Perform the action, get new state and other information
    next_state, reward, done = environment.step(action)
    print(f"Current state: {current_state}, Action: {action}, Next state: {next_state}")
    # Update the current state
    current_state = next_state

    # Additional decision-making logic (like displaying state, pausing, etc.) goes here

