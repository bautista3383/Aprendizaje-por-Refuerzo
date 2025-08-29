# -*- coding: utf-8 -*-
import pygame
import numpy as np
import os
import time
import random
import json
import sys
from datetime import datetime
import subprocess
import traceback

# # --------------------------------------------------------------------------
# --- 1. EXPLICACIÓN DEL CONCEPTO: APRENDIZAJE POR REFUERZO (RL) ---
# --------------------------------------------------------------------------
# El Aprendizaje por Refuerzo es una técnica de Machine Learning donde un
# 'agente' aprende a tomar decisiones interactuando con un 'entorno'.
# Al agente no se le dan las respuestas correctas, sino que aprende por
# prueba y error.
#
# Componentes clave implementados en este juego:
#
# 1. Agente (🤖): Es el avatar del juego (🐭, 🐰, 🐝). Su objetivo es aprender a
#    navegar el laberinto para llegar a la meta.
#
# 2. Entorno (🗺️): Es el laberinto. Contiene paredes, caminos y la meta.
#    El entorno reacciona a las acciones que el agente realiza.
#
# 3. Acción (🕹️): Representa un movimiento que el agente puede ejecutar,
#    como moverse arriba, abajo, izquierda o derecha.
#
# 4. Recompensa (🏆): Es una señal que el entorno le da al agente.
#    - Recompensa grande y positiva: se otorga al llegar a la Salida (S).
#    - Penalización: se aplica al chocar con una pared (#).
#    - Pequeña penalización: se da por cada paso válido para incentivar
#      la exploración. No se le da una recompensa positiva, ya que tiende a formar bucles al conformarse
#      solo en ganar estas pequeñas recompensas sin llegar a la meta.
#      
#      De esta forma el objetivo del juego es perder la menor cantidad de recompensas de 100 que tiene disponiles
#      al llegar a la meta.
#
# El agente utiliza un algoritmo (Q-Learning) para construir un "cerebro"
# o "mapa de calidad" (Q-Table), que le indica la calidad esperada de cada
# acción en cada casilla. Tras muchos intentos (episodios), el agente
# aprende la ruta óptima que maximiza su recompensa total.
# --------------------------------------------------------------------------


# --- CONFIGURACIÓN DEL JUEGO Y DEL APRENDIZAJE ---

# Colores (R, G, B)
COLOR_FONDO = (10, 10, 40)
COLOR_PARED = (20, 80, 150)
COLOR_PASILLO = (200, 200, 220)
COLOR_RASTRO = (255, 255, 0) # Amarillo
COLOR_TEXTO = (255, 255, 255)
COLOR_INFO = (180, 180, 255)

# Parámetros de la ventana
CELL_SIZE = 30
PADDING = 10

# Símbolos (se usarán para la lógica interna)
WALL = "#"
PATH = " "
VISITED_PATH = "*"

# --- AJUSTE DEL SISTEMA DE RECOMPENSAS ---
GOAL_REWARD = 100
WALL_PENALTY = -20
MOVE_REWARD = -0.01

# --- Parámetros del algoritmo Q-Learning ---
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPISODES = 1000

# --- Parámetros de Exploración ---
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# --- CLASE PARA EL LABERINTO (EL ENTORNO) ---

class Maze:
    """Representa el entorno del juego: el laberinto."""
    def __init__(self, maze_data):
        # Limpiar espacios en blanco
        cleaned_data = [row.replace('\xa0', ' ') for row in maze_data]
        max_width = max(len(row) for row in cleaned_data)
        normalized_maze_data = [row.ljust(max_width) for row in cleaned_data]
        
        self.original_maze = np.array([list(row) for row in normalized_maze_data], dtype=str)
        self.height, self.width = self.original_maze.shape
        self.start_pos = self._find_char('E')
        self.goal_pos = self._find_char('S')
        
        if self.start_pos is None or self.goal_pos is None:
            raise ValueError("El laberinto debe contener un punto de Entrada 'E' y uno de Salida 'S'")

        self.maze = np.copy(self.original_maze)
        self.maze[self.start_pos] = PATH
        self.maze[self.goal_pos] = PATH

    def _find_char(self, char):
        for r, row in enumerate(self.original_maze):
            for c, val in enumerate(row):
                if val == char:
                    return (r, c)
        return None

    def get_state_for_pos(self, pos):
        return pos[0] * self.width + pos[1]

    def get_pos_for_state(self, state):
        return (state // self.width, state % self.width)

    def step(self, state, action):
        current_pos = self.get_pos_for_state(state)
        new_pos = self._get_new_pos_from_action(current_pos, action)

        if not (0 <= new_pos[0] < self.height and 0 <= new_pos[1] < self.width and self.maze[new_pos] != WALL):
            return state, WALL_PENALTY, False, "Collision"

        if new_pos == self.goal_pos:
            return self.get_state_for_pos(new_pos), GOAL_REWARD, True, "Goal"
        
        return self.get_state_for_pos(new_pos), MOVE_REWARD, False, "Move"

    def _get_new_pos_from_action(self, pos, action):
        r, c = pos
        if action == 0: r -= 1 # Arriba
        elif action == 1: r += 1 # Abajo
        elif action == 2: c -= 1 # Izquierda
        elif action == 3: c += 1 # Derecha
        return (r, c)

# --- CLASE DEL AGENTE ---

class Agent:
    """Representa al agente que aprende a resolver el laberinto."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.epsilon = EPSILON_START

    def choose_action(self, state, deterministic=False):
        if deterministic:
            return np.argmax(self.q_table[state, :])
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
        self.q_table[state, action] = new_value

    def decay_epsilon(self):
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

# --- GESTOR PRINCIPAL DEL JUEGO (PYGAME) ---

class Game:
    """Controla el flujo del juego, la ventana de Pygame y las estadísticas."""
    def __init__(self):
        pygame.init()
        self.results_path = os.path.join(os.path.expanduser('~'), 'Documents', 'Laberinto_RL_Resultados')
        os.makedirs(self.results_path, exist_ok=True)
        self.data_filepath = os.path.join(self.results_path, "data_laberinto_aprendizaje_refuerzo.json")
        self.load_data()
        
        self.mazes = {
            "Ayuda_al_raton_a_encontrar_su_queso": {
                "agent": "🐭", "goal": "🧀", "layout": [
                    "###################",
                    "E #       #     # #",
                    "# ### ### ### ### #",
                    "#   #   #         #",
                    "# ##### ### ##### #",
                    "# #       # #     #",
                    "# # ######### #####",
                    "#         #       S",
                    "###################"
                ]},
            "Ayuda_al_conejo_a_encontrar_su_zanahoria": {
                "agent": "🐰", "goal": "🥕", "layout": [
                    "###################",
                    "E   #   #         #",
                    "### # # # ### #####",
                    "# # # # # #       #",
                    "# # ### # # #######",
                    "# #       #       #",
                    "# ####### ##### ###",
                    "#             #   S",
                    "###################"
                ]},
            "Ayuda_a_la_abeja_a_encontrar_su_panal": {
                "agent": "🐝", "goal": "🍯", "layout": [
                    "###################",
                    "E #             # #",
                    "# # ##### ##### # #",
                    "#       # #   #   #",
                    "# ##### ### # # ###",
                    "# #   # #   #   # #",
                    "# ### ##### ##### #",
                    "#   #             S",
                    "###################"
                ]}
        }
        self.clock = pygame.time.Clock()
        
        # --- Se crea una lista aleatoria de laberintos para las primeras 3 partidas ---
        self.initial_maze_queue = list(self.mazes.keys())
        random.shuffle(self.initial_maze_queue)

    def setup_window(self, maze_width, maze_height):
        """Configura la ventana de Pygame según el tamaño del laberinto."""
        self.screen_width = maze_width * CELL_SIZE + 2 * PADDING
        self.screen_height = maze_height * CELL_SIZE + 100 
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Laberinto con Aprendizaje por Refuerzo")
        
        try:
            self.emoji_font = pygame.font.Font("C:/Windows/Fonts/seguiemj.ttf", int(CELL_SIZE * 0.7))
        except FileNotFoundError:
            self.emoji_font = pygame.font.Font(None, int(CELL_SIZE * 0.7))
        self.text_font = pygame.font.Font(None, 22)
        self.title_font = pygame.font.Font(None, 32)
        self.subtitle_font = pygame.font.Font(None, 28)

    def load_data(self):
        try:
            with open(self.data_filepath, "r", encoding='utf-8') as f:
                data = json.load(f)
                self.stats = data.get("estadisticas_por_laberinto", {})
                self.game_logs = data.get("historial_partidas", [])
        except (FileNotFoundError, json.JSONDecodeError):
            self.stats = {}
            self.game_logs = []

    def _get_default_maze_stats(self):
        return {
            "partidas_jugadas": 0, 
            "victorias": 0, 
            "pasos_optimizados_por_refuerzo": 0,
            "recompensa_aprendizaje_por_refuerzo": 0
        }

    def save_data(self, new_game_log):
        self.game_logs.append(new_game_log)
        data_to_save = {
            "estadisticas_por_laberinto": self.stats, 
            "historial_partidas": self.game_logs
        }
        with open(self.data_filepath, "w", encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        return self.data_filepath

    def update_stats(self, maze_name, steps, reward):
        maze_stats = self.stats.get(maze_name, self._get_default_maze_stats())

        maze_stats["partidas_jugadas"] += 1
        if reward > 0: 
            maze_stats["victorias"] += 1
        
        maze_stats["pasos_optimizados_por_refuerzo"] = steps
        maze_stats["recompensa_aprendizaje_por_refuerzo"] = reward
        
        self.stats[maze_name] = maze_stats


    def draw_text(self, text, font, color, x, y, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        self.screen.blit(text_surface, text_rect)

    def draw_maze(self, maze):
        for r in range(maze.height):
            for c in range(maze.width):
                rect = pygame.Rect(c * CELL_SIZE + PADDING, r * CELL_SIZE + PADDING, CELL_SIZE, CELL_SIZE)
                if maze.original_maze[r, c] == WALL:
                    pygame.draw.rect(self.screen, COLOR_PARED, rect)
                else:
                    pygame.draw.rect(self.screen, COLOR_PASILLO, rect)

    def draw_game_state(self, maze, maze_name, agent_char, goal_char, agent_pos=None, path=None, bottom_text=""):
        self.screen.fill(COLOR_FONDO)
        self.draw_text(maze_name.replace("_", " "), self.title_font, COLOR_TEXTO, self.screen_width // 2, 30, center=True)
        self.draw_maze(maze)

        if path:
            for pos in path:
                center_x = pos[1] * CELL_SIZE + PADDING + CELL_SIZE // 2
                center_y = pos[0] * CELL_SIZE + PADDING + CELL_SIZE // 2
                pygame.draw.circle(self.screen, COLOR_RASTRO, (center_x, center_y), 5)
        
        self.draw_text(goal_char, self.emoji_font, COLOR_TEXTO, 
                       maze.goal_pos[1] * CELL_SIZE + PADDING + CELL_SIZE // 2, 
                       maze.goal_pos[0] * CELL_SIZE + PADDING + CELL_SIZE // 2, center=True)
        if agent_pos:
            self.draw_text(agent_char, self.emoji_font, COLOR_TEXTO,
                           agent_pos[1] * CELL_SIZE + PADDING + CELL_SIZE // 2,
                           agent_pos[0] * CELL_SIZE + PADDING + CELL_SIZE // 2, center=True)
        
        self.draw_text(bottom_text, self.text_font, COLOR_TEXTO, PADDING, self.screen_height - 30)
        pygame.display.flip()

    def wait_for_key(self):
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE or event.key == pygame.K_KP_ENTER:
                        waiting = False

    def play(self):
        print("Configurando la ventana del menú...")
        self.setup_window(20, 15)
        print("Ventana configurada. Entrando al bucle de menú.")
        
        running = True
        while running:
            self.screen.fill(COLOR_FONDO)
            self.draw_text("Laberinto con Aprendizaje por Refuerzo", self.title_font, COLOR_TEXTO, self.screen_width // 2, 50, center=True)
            self.draw_text("Presiona 'J' para jugar una partida", self.text_font, COLOR_INFO, self.screen_width // 2, 150, center=True)
            self.draw_text("Presiona 'R' para resetear estadísticas", self.text_font, COLOR_INFO, self.screen_width // 2, 180, center=True)
            self.draw_text("Presiona 'ESC' para salir", self.text_font, COLOR_INFO, self.screen_width // 2, 210, center=True)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_j:
                        # --- Lógica para seleccionar el laberinto ---
                        if self.initial_maze_queue:
                            # Si la lista no está vacía, saca el siguiente laberinto
                            maze_name = self.initial_maze_queue.pop(0)
                            maze_info = self.mazes[maze_name]
                        else:
                            # Si la lista está vacía, elige uno al azar independientemente que se repitan
                            maze_name, maze_info = random.choice(list(self.mazes.items()))
                        
                        maze = Maze(maze_info["layout"])
                        self.setup_window(maze.width, maze.height)
                        
                        self.screen.fill(COLOR_FONDO)
                        self.draw_text("Laberinto elegido:", self.subtitle_font, COLOR_TEXTO, self.screen_width // 2, 50, center=True)
                        self.draw_text(maze_name.replace('_', ' '), self.text_font, COLOR_INFO, self.screen_width // 2, 90, center=True)
                        self.draw_text(f"{maze_info['agent']} -> {maze_info['goal']}", self.emoji_font, COLOR_TEXTO, self.screen_width // 2, 130, center=True)
                        self.draw_text("Presiona Enter para continuar...", self.text_font, COLOR_TEXTO, self.screen_width // 2, 200, center=True)
                        pygame.display.flip()
                        self.wait_for_key()

                        self.run_game(maze_info, maze_name)
                        
                        self.setup_window(20, 15)

                    if event.key == pygame.K_r:
                        if os.path.exists(self.data_filepath): os.remove(self.data_filepath)
                        for f in os.listdir(self.results_path):
                            if f.startswith("partida_") and f.endswith(".png"):
                                os.remove(os.path.join(self.results_path, f))
                        self.load_data()
                        # --- Reiniciar la cola de laberintos al borrar datos ---
                        self.initial_maze_queue = list(self.mazes.keys())
                        random.shuffle(self.initial_maze_queue)
                        self.draw_text("¡Datos borrados!", self.text_font, (0, 255, 0), self.screen_width // 2, 250, center=True)
                        pygame.display.flip()
                        time.sleep(2)
                    if event.key == pygame.K_ESCAPE:
                        running = False
        pygame.quit()
        sys.exit()

    def run_single_episode(self, maze, agent, deterministic, maze_name, agent_char, goal_char, show_simulation=False):
        state = maze.get_state_for_pos(maze.start_pos)
        done = False
        steps, total_reward = 0, 0
        path_taken = [maze.start_pos]
        log_entries = []
        max_steps = maze.width * maze.height * 2

        while not done and steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()

            action = agent.choose_action(state, deterministic)
            current_pos = maze.get_pos_for_state(state)
            
            next_state, reward, done, outcome = maze.step(state, action)
            
            if not deterministic:
                agent.learn(state, action, reward, next_state)
            
            state = next_state
            next_pos = maze.get_pos_for_state(state)
            total_reward += reward
            steps += 1
            if next_pos not in path_taken: path_taken.append(next_pos)

            log_entry = ""
            base_text = f"Paso {steps}: Desde {current_pos}, acción '{['↑','↓','←','→'][action]}'."
            
            if outcome == "Collision":
                log_entry = f"{base_text} Resultado: tope, Penalización: {reward:+.2f}, Total acumulado: {total_reward:+.2f}"
            elif outcome == "Goal":
                log_entry = f"{base_text} Resultado: meta-salida, Recompensa: {reward:+.2f}, Total acumulado: {total_reward:+.2f}"
            else: # Move
                log_entry = f"{base_text} Resultado: movimiento libre, Pequeña Penalización: {reward:+.2f}, Total acumulado: {total_reward:+.2f}"

            log_entries.append(log_entry)
            
            if show_simulation:
                self.draw_game_state(maze, maze_name, agent_char, goal_char, agent_pos=next_pos, path=path_taken, bottom_text=log_entry)
                self.clock.tick(10)

        return {"outcome": "Meta alcanzada" if done else "Límite de pasos", "steps": steps, "total_reward": round(total_reward, 2), "path": path_taken, "log": log_entries}

    def run_game(self, maze_info, maze_name):
        maze_data = maze_info["layout"]
        agent_char = maze_info["agent"]
        goal_char = maze_info["goal"]
        
        maze = Maze(maze_data)
        agent = Agent(state_size=maze.width * maze.height, action_size=4)
        learning_log = {}
        
        episodes_to_log = [1, 50, 200]
        
        for episode in range(1, EPISODES + 1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if episode in episodes_to_log:
                self.screen.fill(COLOR_FONDO)
                self.draw_text(f"Mostrando Intento de Aprendizaje No. {episode}", self.subtitle_font, COLOR_TEXTO, self.screen_width // 2, 50, center=True)
                self.draw_text(maze_name.replace('_', ' '), self.text_font, COLOR_INFO, self.screen_width // 2, 80, center=True)
                self.draw_text(f"{agent_char} -> {goal_char}", self.emoji_font, COLOR_TEXTO, self.screen_width // 2, 110, center=True)
                pygame.display.flip()
                time.sleep(3)
                
                temp_agent = Agent(agent.state_size, agent.action_size)
                temp_agent.q_table = np.copy(agent.q_table)
                temp_agent.epsilon = agent.epsilon
                log_result = self.run_single_episode(maze, temp_agent, False, maze_name, agent_char, goal_char, show_simulation=True)
                
                if 'path' in log_result:
                    del log_result['path']
                learning_log[f"intento_{episode}"] = log_result
                
                self.screen.fill(COLOR_FONDO)
                self.draw_text("Intento finalizado. Presiona Enter para continuar.", self.text_font, COLOR_INFO, self.screen_width // 2, 100, center=True)
                pygame.display.flip()
                self.wait_for_key()

            state = maze.get_state_for_pos(maze.start_pos)
            done = False
            max_steps = maze.width * maze.height * 2 
            steps = 0
            while not done and steps < max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                action = agent.choose_action(state)
                next_state, reward, done, _ = maze.step(state, action)
                agent.learn(state, action, reward, next_state)
                state = next_state
                steps += 1 
            
            agent.decay_epsilon()
            
            if episode % (EPISODES // 20) == 0:
                self.screen.fill(COLOR_FONDO)
                self.draw_text("El agente está aprendiendo...", self.subtitle_font, COLOR_TEXTO, self.screen_width // 2, 50, center=True)
                self.draw_text(f"Progreso: {episode/EPISODES*100:.0f}%", self.text_font, COLOR_INFO, self.screen_width // 2, 80, center=True)
                pygame.display.flip()

        self.screen.fill(COLOR_FONDO)
        self.draw_text("¡Aprendizaje completado!", self.title_font, COLOR_TEXTO, self.screen_width // 2, 50, center=True)
        self.draw_text("Mostrando solución final...", self.subtitle_font, COLOR_INFO, self.screen_width // 2, 90, center=True)
        self.draw_text(maze_name.replace('_', ' '), self.text_font, COLOR_INFO, self.screen_width // 2, 120, center=True)
        self.draw_text(f"{agent_char} -> {goal_char}", self.emoji_font, COLOR_TEXTO, self.screen_width // 2, 150, center=True)
        pygame.display.flip()
        time.sleep(3)

        final_run_data = self.run_single_episode(maze, agent, True, maze_name, agent_char, goal_char, show_simulation=True)
        
        partida_num = sum(s.get('partidas_jugadas', 0) for s in self.stats.values()) + 1
        
        self.update_stats(maze_name, final_run_data["steps"], final_run_data["total_reward"])
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_filename = f"partida_{partida_num}_{maze_name.replace(' ', '_')}_{timestamp}.png"
        img_path = os.path.join(self.results_path, img_filename)
        pygame.image.save(self.screen, img_path)
        print(f"\nResultado guardado como imagen en: {img_path}")

        current_game_log = {
            "partida_numero": partida_num,
            "laberinto": maze_name,
            "fecha_hora": timestamp,
            "resumen_resultado": {
                "resultado": final_run_data['outcome'],
                "pasos_finales": final_run_data['steps'],
                "recompensa_final": final_run_data['total_reward']
            },
            "detalle_aprendizaje": learning_log
        }
        log_path = self.save_data(current_game_log)
        
        end_screen_running = True
        while end_screen_running:
            self.screen.fill(COLOR_FONDO)
            self.draw_text("¡Partida Finalizada!", self.title_font, COLOR_TEXTO, self.screen_width // 2, 50, center=True)
            self.draw_text(f"Resultado: {final_run_data['outcome']}", self.text_font, COLOR_INFO, self.screen_width // 2, 100, center=True)
            self.draw_text(f"Pasos: {final_run_data['steps']} | Recompensa: {final_run_data['total_reward']}", self.text_font, COLOR_INFO, self.screen_width // 2, 130, center=True)
            self.draw_text("Presiona 'J' para jugar de nuevo", self.text_font, COLOR_TEXTO, self.screen_width // 2, 200, center=True)
            self.draw_text("Presiona 'ESC' para salir", self.text_font, COLOR_TEXTO, self.screen_width // 2, 230, center=True)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_j:
                        end_screen_running = False 
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

        try:
            if sys.platform == "win32":
                os.startfile(os.path.abspath(img_path))
                os.startfile(os.path.abspath(log_path))
            elif sys.platform == "darwin":
                subprocess.run(["open", os.path.abspath(img_path)], check=True)
                subprocess.run(["open", os.path.abspath(log_path)], check=True)
            else:
                subprocess.run(["xdg-open", os.path.abspath(img_path)], check=True)
                subprocess.run(["xdg-open", os.path.abspath(log_path)], check=True)
        except Exception as e:
            print(f"No se pudo abrir el archivo de log automáticamente: {e}")

# --- Iniciar el Juego ---
if __name__ == "__main__":
    try:
        print("Inicializando el juego...")
        game = Game()
        print("Iniciando el bucle principal del juego...")
        game.play()
    except Exception as e:
        print("\n--- OCURRIÓ UN ERROR INESPERADO ---")
        traceback.print_exc()
        print("------------------------------------")
        input("\nEl programa ha finalizado debido a un error. Presiona Enter para cerrar.")
