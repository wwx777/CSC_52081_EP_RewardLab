from collections import deque

from rewards.base_reward import BaseReward, register_reward


@register_reward("shortest_path_progress")
class ShortestPathProgressReward(BaseReward):
    """
    用迷宫中的真实最短路距离做 shaping。
    这比 Manhattan distance 更适合 maze。
    """

    def __init__(self):
        self.prev_distance = None
        self.prev_pos = None
        self._dist_cache = None
        self._goal_pos = None

    def reset(self, agent_pos, goal_pos, maze):
        self._goal_pos = goal_pos
        self._dist_cache = self._build_distance_map(goal_pos, maze)
        self.prev_distance = self._get_dist(agent_pos)
        self.prev_pos = agent_pos

    def compute(
      self,
      agent_pos,
      goal_pos,
      reached_goal,
      visited,
      steps,
      maze,
      **kwargs
  ):
      reward = -0.01

      current_distance = self._get_dist(agent_pos)

      if self.prev_distance is not None and current_distance is not None:
          progress = self.prev_distance - current_distance
          reward += 1.0 * progress

      if agent_pos == self.prev_pos and not reached_goal:
          reward -= 0.05

      if agent_pos in visited and agent_pos != self.prev_pos and not reached_goal:
          reward -= 0.01

      if reached_goal:
          reward += 20.0

      self.prev_distance = current_distance
      self.prev_pos = agent_pos

      return float(reward)

    def _get_dist(self, pos):
        if self._dist_cache is None:
            return None
        return self._dist_cache.get(pos, None)

    def _build_distance_map(self, goal_pos, maze):
        """
        从 goal 反向 BFS，得到每个可走格到 goal 的真实最短路距离。
        """
        h, w = maze.shape
        q = deque()
        dist = {}

        if maze[goal_pos[0], goal_pos[1]] != 0:
            return dist

        q.append(goal_pos)
        dist[goal_pos] = 0

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while q:
            x, y = q.popleft()
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and maze[nx, ny] == 0:
                    if (nx, ny) not in dist:
                        dist[(nx, ny)] = dist[(x, y)] + 1
                        q.append((nx, ny))

        return dist