import math
import sys
from typing import Tuple, Optional
import threading
import time

import pygame
import numpy as np

from engine.robot_backend import Planar3RRobot, FKResult, IKResult
from engine.workspace import WorkspaceSampler


WINDOW_SIZE = (1000, 700)
WORLD_SCALE = 220.0  # pixels per meter (approx)
ORIGIN_SCREEN = np.array([WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2 + 150], dtype=float)

# Color theme
COLOR_BG = (20, 22, 28)
COLOR_BG_DARK = (15, 17, 23)
COLOR_PANEL = (30, 33, 42)
COLOR_PANEL_BORDER = (60, 65, 80)
COLOR_TEXT_PRIMARY = (240, 245, 255)
COLOR_TEXT_SECONDARY = (180, 190, 210)
COLOR_TEXT_MUTED = (120, 130, 150)
COLOR_AXIS_X = (220, 80, 80)
COLOR_AXIS_Y = (80, 200, 120)
COLOR_ROBOT_LINK = (100, 180, 255)
COLOR_ROBOT_JOINT = (255, 200, 100)
COLOR_ROBOT_EE = (255, 255, 255)
COLOR_WORKSPACE = (80, 160, 220)
COLOR_MANIPULABILITY = (120, 220, 160)


def world_to_screen(p: np.ndarray) -> Tuple[int, int]:
    x, y = p[0], p[1]
    sx = ORIGIN_SCREEN[0] + x * WORLD_SCALE
    sy = ORIGIN_SCREEN[1] - y * WORLD_SCALE
    return int(round(sx)), int(round(sy))


def draw_axes(surface: pygame.Surface) -> None:
    # X axis (red), Y axis (green) - with subtle glow
    x_axis_y = int(ORIGIN_SCREEN[1])
    y_axis_x = int(ORIGIN_SCREEN[0])
    
    # Draw main axes with glow effect
    pygame.draw.line(surface, COLOR_AXIS_X, (0, x_axis_y), (WINDOW_SIZE[0], x_axis_y), 2)
    pygame.draw.line(surface, COLOR_AXIS_Y, (y_axis_x, 0), (y_axis_x, WINDOW_SIZE[1]), 2)
    
    # scale bar + ticks
    font = pygame.font.SysFont("Arial", 16, bold=True)
    m_len = 0.5
    start = world_to_screen(np.array([-0.6, -0.05]))
    end = world_to_screen(np.array([-0.1, -0.05]))
    pygame.draw.line(surface, COLOR_TEXT_PRIMARY, start, end, 3)
    pygame.draw.circle(surface, COLOR_TEXT_PRIMARY, start, 3)
    pygame.draw.circle(surface, COLOR_TEXT_PRIMARY, end, 3)
    surface.blit(font.render(f"{m_len:.1f} m", True, COLOR_TEXT_SECONDARY), (start[0], start[1] - 22))
    
    # axis ticks every 0.1 m within screen bounds
    tick_color = COLOR_TEXT_MUTED
    label_color = COLOR_TEXT_SECONDARY
    # X ticks
    x_min_w = -(ORIGIN_SCREEN[0]) / WORLD_SCALE
    x_max_w = (WINDOW_SIZE[0] - ORIGIN_SCREEN[0]) / WORLD_SCALE
    t = math.ceil(x_min_w / 0.1) * 0.1
    while t <= x_max_w:
        sx, sy = world_to_screen(np.array([t, 0.0]))
        pygame.draw.line(surface, tick_color, (sx, x_axis_y - 5), (sx, x_axis_y + 5), 2)
        if abs(t) < 1e-9 or abs((t*10) % 5) > 1e-6:
            # fewer labels; only every 0.5 m
            pass
        else:
            label_font = pygame.font.SysFont("Arial", 14)
            surface.blit(label_font.render(f"{t:.1f}", True, label_color), (sx - 12, x_axis_y + 8))
        t += 0.1
    # Y ticks
    y_min_w = -(WINDOW_SIZE[1] - ORIGIN_SCREEN[1]) / WORLD_SCALE
    y_max_w = (ORIGIN_SCREEN[1]) / WORLD_SCALE
    t = math.ceil(y_min_w / 0.1) * 0.1
    while t <= y_max_w:
        sx, sy = world_to_screen(np.array([0.0, t]))
        pygame.draw.line(surface, tick_color, (y_axis_x - 5, sy), (y_axis_x + 5, sy), 2)
        if abs(t) < 1e-9 or abs((t*10) % 5) > 1e-6:
            pass
        else:
            label_font = pygame.font.SysFont("Arial", 14)
            surface.blit(label_font.render(f"{t:.1f}", True, label_color), (y_axis_x + 8, sy - 10))
        t += 0.1


def draw_robot(surface: pygame.Surface, robot: Planar3RRobot, q: np.ndarray) -> None:
    points = robot.link_endpoints(q)
    
    # Draw links with gradient effect (shadow first)
    for i in range(len(points) - 1):
        a = world_to_screen(points[i])
        b = world_to_screen(points[i + 1])
        
        # Shadow for depth
        shadow_offset = 2
        pygame.draw.line(surface, (20, 20, 30), 
                        (a[0] + shadow_offset, a[1] + shadow_offset),
                        (b[0] + shadow_offset, b[1] + shadow_offset), 8)
        
        # Main link with gradient-like effect
        pygame.draw.line(surface, COLOR_ROBOT_LINK, a, b, 8)
        # Highlight on top
        pygame.draw.line(surface, (150, 220, 255), a, b, 3)
    
    # Draw joints with glow effect
    for i in range(len(points) - 1):
        joint = world_to_screen(points[i])
        # Outer glow
        pygame.draw.circle(surface, (255, 200, 80), joint, 12, 2)
        # Main joint
        pygame.draw.circle(surface, COLOR_ROBOT_JOINT, joint, 10)
        # Inner highlight
        pygame.draw.circle(surface, (255, 230, 150), joint, 6)
        # Center dot
        pygame.draw.circle(surface, (255, 250, 200), joint, 3)
    
    # End-effector with special styling
    ee = world_to_screen(points[-1])
    # Outer glow
    pygame.draw.circle(surface, (240, 240, 250), ee, 12, 2)
    # Main end-effector
    pygame.draw.circle(surface, COLOR_ROBOT_EE, ee, 10)
    # Inner highlight
    pygame.draw.circle(surface, (220, 220, 255), ee, 6)
    # Center dot
    pygame.draw.circle(surface, (200, 200, 255), ee, 4)


def draw_workspace(surface: pygame.Surface, samples: np.ndarray) -> None:
    if samples.size == 0:
        return
    pts = [world_to_screen(p) for p in samples]
    for p in pts:
        # Draw workspace points with subtle glow
        pygame.draw.circle(surface, COLOR_WORKSPACE, p, 2)
        pygame.draw.circle(surface, (150, 220, 255), p, 1)

def draw_reach_boundary(surface: pygame.Surface, robot: Planar3RRobot) -> None:
    L = robot.link_lengths
    r_max = sum(L)
    r_min = max(0.0, abs(L[0] - (L[1] + L[2])))
    center = (int(ORIGIN_SCREEN[0]), int(ORIGIN_SCREEN[1]))
    
    # Draw outer and inner circle as boundaries with dashed effect
    color_outer = COLOR_WORKSPACE
    color_inner = (220, 140, 140)
    radius_outer = int(r_max * WORLD_SCALE)
    radius_inner = int(r_min * WORLD_SCALE)
    
    # Draw dashed circles
    if radius_outer > 0:
        for angle in range(0, 360, 5):
            if angle % 15 < 8:  # Create dashed effect
                rad = math.radians(angle)
                start = (center[0] + int(radius_outer * math.cos(rad)), 
                        center[1] + int(radius_outer * math.sin(rad)))
                end_rad = math.radians(angle + 5)
                end = (center[0] + int(radius_outer * math.cos(end_rad)),
                      center[1] + int(radius_outer * math.sin(end_rad)))
                pygame.draw.line(surface, color_outer, start, end, 2)
    
    if radius_inner > 0 and radius_inner != radius_outer:
        for angle in range(0, 360, 5):
            if angle % 15 < 8:
                rad = math.radians(angle)
                start = (center[0] + int(radius_inner * math.cos(rad)),
                        center[1] + int(radius_inner * math.sin(rad)))
                end_rad = math.radians(angle + 5)
                end = (center[0] + int(radius_inner * math.cos(end_rad)),
                      center[1] + int(radius_inner * math.sin(end_rad)))
                pygame.draw.line(surface, color_inner, start, end, 2)
    
    # Labels with background
    font = pygame.font.SysFont("Arial", 16, bold=True)
    if radius_outer > 0:
        label_text = f"r_max={r_max:.2f} m"
        text_surface = font.render(label_text, True, COLOR_TEXT_PRIMARY)
        text_rect = text_surface.get_rect()
        bg_rect = text_rect.copy()
        bg_rect.x = center[0] + radius_outer - 90
        bg_rect.y = center[1] - 20
        bg_rect.inflate_ip(8, 4)
        pygame.draw.rect(surface, COLOR_PANEL, bg_rect)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, bg_rect, 1)
        surface.blit(text_surface, (bg_rect.x + 4, bg_rect.y + 2))
    
    if radius_inner > 0 and radius_inner != radius_outer:
        label_text = f"r_min={r_min:.2f} m"
        text_surface = font.render(label_text, True, COLOR_TEXT_PRIMARY)
        text_rect = text_surface.get_rect()
        bg_rect = text_rect.copy()
        bg_rect.x = center[0] + radius_inner - 90
        bg_rect.y = center[1] + 4
        bg_rect.inflate_ip(8, 4)
        pygame.draw.rect(surface, COLOR_PANEL, bg_rect)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, bg_rect, 1)
        surface.blit(text_surface, (bg_rect.x + 4, bg_rect.y + 2))

def draw_manipulability(surface: pygame.Surface, ee_pos_w: np.ndarray, U: np.ndarray, S: np.ndarray) -> None:
    """
    Visualize task-space principal directions (columns of U) scaled by singular values S at the EE.
    """
    # scale for on-screen visibility (meters -> pixels via WORLD_SCALE)
    scale = 0.25  # meters per unit singular value for vector length
    origin_px = world_to_screen(ee_pos_w)
    colors = [(100, 240, 140), (240, 200, 120)]
    for i in range(min(2, U.shape[1])):
        direction_w = U[:, i]  # 2-vector in world XY
        length_m = float(S[i]) * scale
        tip_w = ee_pos_w + direction_w * length_m
        tip_px = world_to_screen(tip_w)
        # Draw with glow
        pygame.draw.line(surface, colors[i], origin_px, tip_px, 4)
        pygame.draw.circle(surface, colors[i], tip_px, 5)
        pygame.draw.circle(surface, (255, 255, 255), tip_px, 2)
    # Ellipse outline
    pts = []
    for t in np.linspace(0, 2*np.pi, 40, endpoint=True):
        # ellipse param in principal frame then map to world: ee + U * (S * [cos t, sin t])
        local = np.array([np.cos(t), np.sin(t)], dtype=float) * (S[:2] * scale)
        world = ee_pos_w + U[:, :2] @ local
        pts.append(world_to_screen(world))
    if len(pts) >= 2:
        pygame.draw.aalines(surface, (140, 220, 250), True, pts, 2)

class MotionController:
    def __init__(self):
        self.active = False
        self.q_start = np.zeros(3, dtype=float)
        self.q_end = np.zeros(3, dtype=float)
        self.t_start = 0.0
        self.duration = 0.5  # seconds

    def start(self, q_current: np.ndarray, q_target: np.ndarray, duration: float = 0.5):
        self.q_start = np.array(q_current, dtype=float)
        self.q_end = np.array(q_target, dtype=float)
        self.t_start = time.perf_counter()
        self.duration = max(0.05, float(duration))
        self.active = True

    def is_active(self) -> bool:
        return self.active

    def update(self) -> np.ndarray:
        if not self.active:
            return self.q_end
        t = (time.perf_counter() - self.t_start) / self.duration
        if t >= 1.0:
            self.active = False
            return self.q_end
        # smoothstep easing
        u = t * t * (3 - 2 * t)
        q = (1 - u) * self.q_start + u * self.q_end
        return q


class TextInput:
    def __init__(self, font: pygame.font.Font):
        self.font = font
        self.active = False
        self.text = ""
        self.prompt = ""
        self.color_bg = COLOR_PANEL
        self.color_border = COLOR_ROBOT_LINK
        self.color_text = COLOR_TEXT_PRIMARY
        self.color_prompt = COLOR_TEXT_SECONDARY
        self.on_submit = None  # type: Optional[callable]

    def start(self, prompt: str, on_submit):
        self.active = True
        self.text = ""
        self.prompt = prompt
        self.on_submit = on_submit

    def cancel(self):
        self.active = False
        self.text = ""
        self.prompt = ""
        self.on_submit = None

    def handle_event(self, event: pygame.event.Event):
        if not self.active:
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.cancel()
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                if self.on_submit is not None:
                    try:
                        self.on_submit(self.text)
                    finally:
                        self.cancel()
            else:
                ch = event.unicode
                if ch and ch.isprintable():
                    self.text += ch

    def draw(self, surface: pygame.Surface):
        if not self.active:
            return
        w, h = WINDOW_SIZE
        box_w, box_h = int(w * 0.85), 80
        x = (w - box_w) // 2
        y = h - box_h - 20
        
        # Draw background with rounded corners effect
        pygame.draw.rect(surface, self.color_bg, (x, y, box_w, box_h))
        pygame.draw.rect(surface, self.color_border, (x, y, box_w, box_h), 3)
        
        # Draw shadow
        shadow_offset = 4
        shadow_rect = pygame.Rect(x + shadow_offset, y + shadow_offset, box_w, box_h)
        shadow_surf = pygame.Surface((box_w, box_h))
        shadow_surf.set_alpha(60)
        shadow_surf.fill((0, 0, 0))
        surface.blit(shadow_surf, (x + shadow_offset, y + shadow_offset))
        
        # Draw content
        prompt_font = pygame.font.SysFont("Arial", 16, bold=True)
        text_font = pygame.font.SysFont("Arial", 20)
        
        prompt_surf = prompt_font.render(self.prompt, True, self.color_prompt)
        text_surf = text_font.render(self.text + "|", True, self.color_text)
        
        # Draw with padding
        padding = 16
        surface.blit(prompt_surf, (x + padding, y + 12))
        
        # Draw text input area with border
        text_y = y + 42
        pygame.draw.line(surface, COLOR_PANEL_BORDER, (x + padding, text_y - 2), (x + box_w - padding, text_y - 2), 2)
        surface.blit(text_surf, (x + padding, text_y))


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Planar 3R Robot — Pygame + Robotic Toolbox")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    input_ui = TextInput(font)
    motion = MotionController()

    robot = Planar3RRobot([0.25, 0.25, 0.18])
    sampler = WorkspaceSampler(robot)

    q = np.array([0.0, 0.0, 0.0])
    ik_target: Optional[np.ndarray] = None
    ik_reachable: Optional[bool] = None
    ws_points = np.empty((0, 2))
    ws_loading = False
    ws_lock = threading.Lock()
    show_workspace = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN and not input_ui.active:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit(0)
                if event.key == pygame.K_3:
                    show_workspace = not show_workspace
                    if not show_workspace:
                        with ws_lock:
                            ws_points = np.empty((0, 2))
                        ws_loading = False
                if event.key == pygame.K_r:
                    q[:] = 0.0
                    ik_target = None
                    ik_reachable = None
                if event.key == pygame.K_a:
                    q_target = q.copy(); q_target[0] += math.radians(3)
                    motion.start(q, (q_target + np.pi) % (2*np.pi) - np.pi)
                if event.key == pygame.K_z:
                    q_target = q.copy(); q_target[0] -= math.radians(3)
                    motion.start(q, (q_target + np.pi) % (2*np.pi) - np.pi)
                if event.key == pygame.K_s:
                    q_target = q.copy(); q_target[1] += math.radians(3)
                    motion.start(q, (q_target + np.pi) % (2*np.pi) - np.pi)
                if event.key == pygame.K_x:
                    q_target = q.copy(); q_target[1] -= math.radians(3)
                    motion.start(q, (q_target + np.pi) % (2*np.pi) - np.pi)
                if event.key == pygame.K_d:
                    q_target = q.copy(); q_target[2] += math.radians(3)
                    motion.start(q, (q_target + np.pi) % (2*np.pi) - np.pi)
                if event.key == pygame.K_c:
                    q_target = q.copy(); q_target[2] -= math.radians(3)
                    motion.start(q, (q_target + np.pi) % (2*np.pi) - np.pi)
                # Flip Configuration (สลับท่าทาง: พับแขนขึ้น/ลง)
                if event.key == pygame.K_u:  # Flip Configuration Up (พับแขนขึ้น)
                    fk = robot.fk(q)
                    current_pos = fk.position
                    # หา alternative configuration แบบ elbow up
                    ik = robot.ik_flip_configuration(current_pos, q, prefer_elbow_up=True)
                    if ik.success:
                        motion.start(q, ik.q, duration=0.6)
                if event.key == pygame.K_m:  # Flip Configuration Down (พับแขนลง)
                    fk = robot.fk(q)
                    current_pos = fk.position
                    # หา alternative configuration แบบ elbow down
                    ik = robot.ik_flip_configuration(current_pos, q, prefer_elbow_up=False)
                    if ik.success:
                        motion.start(q, ik.q, duration=0.6)
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    robot.scale_links(1.05)
                    ws_points = np.empty((0, 2))
                if event.key == pygame.K_MINUS:
                    robot.scale_links(1 / 1.05)
                    ws_points = np.empty((0, 2))
                # Start input modes
                if event.key == pygame.K_j:  # joints input
                    def submit_joints(text: str):
                        vals = [t for t in text.replace(" ", "").split(",") if t]
                        if len(vals) == 3:
                            try:
                                qd = np.radians(np.array(list(map(float, vals)), dtype=float))
                                qd = (qd + np.pi) % (2 * np.pi) - np.pi
                                motion.start(q, qd, duration=0.6)
                            except Exception:
                                pass
                    input_ui.start("Enter joints q1,q2,q3 in degrees (e.g. 0,30,-45). Enter=OK, Esc=Cancel", submit_joints)
                if event.key == pygame.K_t:  # target input
                    def submit_target(text: str):
                        vals = [t for t in text.replace(" ", "").split(",") if t]
                        if len(vals) == 2:
                            try:
                                x, y = map(float, vals)
                                target_xy = np.array([x, y], dtype=float)
                                reachable = robot.within_reach(target_xy)
                                nonlocal ik_target, ik_reachable
                                ik_target = np.array([x, y, 0.0], dtype=float)
                                ik_reachable = bool(reachable)
                                ik = robot.ik(target_xy, None)
                                if ik.success:
                                    motion.start(q, ik.q, duration=0.8)
                            except Exception:
                                pass
                    input_ui.start("Enter target x,y in meters (e.g. 0.3,0.1). Enter=OK, Esc=Cancel", submit_target)
                if event.key == pygame.K_l:  # link lengths input
                    def submit_links(text: str):
                        vals = [t for t in text.replace(" ", "").split(",") if t]
                        if len(vals) == 3:
                            try:
                                L = list(map(float, vals))
                                if all(v > 0 for v in L):
                                    robot.link_lengths = L
                                    ws_points[:] = np.empty((0, 2))
                            except Exception:
                                pass
                    input_ui.start("Enter link lengths L1,L2,L3 in meters (e.g. 0.25,0.25,0.18). Enter=OK, Esc=Cancel", submit_links)
            # Forward key events to input UI (if active)
            input_ui.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                wx = (mx - ORIGIN_SCREEN[0]) / WORLD_SCALE
                wy = (ORIGIN_SCREEN[1] - my) / WORLD_SCALE
                target_xy = np.array([wx, wy], dtype=float)
                ik_target = np.array([wx, wy, 0.0], dtype=float)
                ik_reachable = bool(robot.within_reach(target_xy))
                ik = robot.ik(target_xy, None)
                if ik.success:
                    motion.start(q, ik.q, duration=0.8)

        # background with subtle gradient
        screen.fill(COLOR_BG_DARK)
        # Draw subtle grid
        grid_color = (25, 28, 35)
        grid_spacing = 50
        for x in range(0, WINDOW_SIZE[0], grid_spacing):
            pygame.draw.line(screen, grid_color, (x, 0), (x, WINDOW_SIZE[1]), 1)
        for y in range(0, WINDOW_SIZE[1], grid_spacing):
            pygame.draw.line(screen, grid_color, (0, y), (WINDOW_SIZE[0], y), 1)
        
        draw_axes(screen)

        # update motion
        if motion.is_active():
            q = motion.update()
        # compute FK and singularity
        fk: FKResult = robot.fk(q)
        sigma_min, cond = robot.singularity_metrics(q)
        near_singular = (sigma_min < 5e-3) or (cond > 1e5)
        U, S, Vt = robot.svd_jacobian(q)
        
        # Calculate manipulability measures
        # Manipulability measure: w = σ1 * σ2 (product of singular values)
        manipulability = float(S[0] * S[1]) if S.size >= 2 else 0.0
        # Normalized manipulability (0-1 scale for display)
        # Using approximate max for typical 3R robot configuration
        max_manipulability_approx = sum(robot.link_lengths) ** 2 / 4.0  # rough estimate
        normalized_manip = min(1.0, manipulability / max_manipulability_approx) if max_manipulability_approx > 0 else 0.0

        # workspace sampling on-demand (background)
        if show_workspace:
            if ws_points.size == 0 and not ws_loading:
                ws_loading = True
                def _sample_ws():
                    pts = sampler.sample_points(samples_per_joint=60)
                    with ws_lock:
                        nonlocal ws_points, ws_loading
                        ws_points = pts
                        ws_loading = False
                threading.Thread(target=_sample_ws, daemon=True).start()
        else:
            with ws_lock:
                ws_points = np.empty((0, 2))

        if show_workspace:
            draw_reach_boundary(screen, robot)
            draw_workspace(screen, ws_points)
        draw_reach_boundary(screen, robot)
        draw_robot(screen, robot, q)
        # draw manipulability at EE
        try:
            draw_manipulability(screen, fk.position, U, S)
        except Exception:
            pass

        # IK target visualization
        if ik_target is not None:
            target_px = world_to_screen(ik_target[:2])
            if ik_reachable:
                # Draw target with glow effect
                pygame.draw.circle(screen, (80, 200, 120), target_px, 10, 2)
                pygame.draw.circle(screen, (100, 255, 140), target_px, 8)
                pygame.draw.circle(screen, (200, 255, 220), target_px, 5)
                pygame.draw.circle(screen, (255, 255, 255), target_px, 3)
            else:
                # draw red cross if unreachable with glow
                cross_size = 8
                cross_thick = 3
                # Outer glow
                pygame.draw.line(screen, (200, 80, 80),
                               (target_px[0] - cross_size - 2, target_px[1] - cross_size - 2),
                               (target_px[0] + cross_size + 2, target_px[1] + cross_size + 2),
                               cross_thick + 2)
                pygame.draw.line(screen, (200, 80, 80),
                               (target_px[0] - cross_size - 2, target_px[1] + cross_size + 2),
                               (target_px[0] + cross_size + 2, target_px[1] - cross_size - 2),
                               cross_thick + 2)
                # Main cross
                pygame.draw.line(screen, (255, 100, 100),
                               (target_px[0] - cross_size, target_px[1] - cross_size),
                               (target_px[0] + cross_size, target_px[1] + cross_size),
                               cross_thick)
                pygame.draw.line(screen, (255, 100, 100),
                               (target_px[0] - cross_size, target_px[1] + cross_size),
                               (target_px[0] + cross_size, target_px[1] - cross_size),
                               cross_thick)

        # End-effector label near tool with background
        ee_px = world_to_screen(fk.position)
        label_font = pygame.font.SysFont("Arial", 14, bold=True)
        ee_label = label_font.render(f"EE=({fk.position[0]:.3f}, {fk.position[1]:.3f}) m", True, COLOR_TEXT_PRIMARY)
        label_rect = ee_label.get_rect()
        bg_rect = label_rect.copy()
        bg_rect.x = ee_px[0] + 10
        bg_rect.y = ee_px[1] - 24
        bg_rect.inflate_ip(8, 4)
        pygame.draw.rect(screen, COLOR_PANEL, bg_rect)
        pygame.draw.rect(screen, COLOR_PANEL_BORDER, bg_rect, 1)
        screen.blit(ee_label, (bg_rect.x + 4, bg_rect.y + 2))

        # HUD with styled panels
        hud_padding = 12
        hud_margin = 16
        hud_x = hud_margin
        hud_y = hud_margin
        
        # Title
        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        title_text = title_font.render("Planar 3R Robot Control", True, COLOR_TEXT_PRIMARY)
        title_bg = pygame.Rect(hud_x, hud_y, title_text.get_width() + hud_padding * 2, title_text.get_height() + hud_padding)
        pygame.draw.rect(screen, COLOR_PANEL, title_bg)
        pygame.draw.rect(screen, COLOR_PANEL_BORDER, title_bg, 2)
        screen.blit(title_text, (hud_x + hud_padding, hud_y + hud_padding // 2))
        hud_y += title_bg.height + 8
        
        # Status panel
        status_lines = [
            f"Joint Angles: ({np.degrees(q[0]):.1f}°, {np.degrees(q[1]):.1f}°, {np.degrees(q[2]):.1f}°)",
            f"End-Effector: ({fk.position[0]:.3f}, {fk.position[1]:.3f}) m",
            f"Yaw: {np.degrees(fk.yaw):.1f}°",
            f"Singularity: {'⚠ YES' if fk.singular else '✓ no'}", 
            f"σ_min: {sigma_min:.4e}  cond: {cond:.2e}",
        ]
        
        # Manipulability panel
        manipulability_lines = [
            "Manipulability:",
            f"  Measure: {manipulability:.6f}",
            f"  Normalized: {normalized_manip:.1%}",
            f"  σ₁: {S[0]:.4f}  σ₂: {S[1]:.4f}",
            f"  Ellipse Area: {manipulability * np.pi:.6f}",
        ]
        max_width = max([font.size(line)[0] for line in status_lines]) + hud_padding * 2
        status_bg = pygame.Rect(hud_x, hud_y, max_width, len(status_lines) * 22 + hud_padding * 2)
        pygame.draw.rect(screen, COLOR_PANEL, status_bg)
        pygame.draw.rect(screen, COLOR_PANEL_BORDER, status_bg, 2)
        for i, line in enumerate(status_lines):
            color = (255, 150, 150) if '⚠' in line else COLOR_TEXT_SECONDARY
            text_surf = font.render(line, True, color)
            screen.blit(text_surf, (hud_x + hud_padding, hud_y + hud_padding + i * 22))
        hud_y += status_bg.height + 8
        
        # Manipulability panel
        max_width_manip = max([font.size(line)[0] for line in manipulability_lines]) + hud_padding * 2
        manip_bg = pygame.Rect(hud_x, hud_y, max_width_manip, len(manipulability_lines) * 20 + hud_padding * 2)
        pygame.draw.rect(screen, COLOR_PANEL, manip_bg)
        pygame.draw.rect(screen, COLOR_PANEL_BORDER, manip_bg, 2)
        
        # Color code manipulability (green = good, yellow = medium, red = low)
        for i, line in enumerate(manipulability_lines):
            if i == 0:  # Header
                color = COLOR_TEXT_PRIMARY
            elif "Normalized" in line:
                # Color code based on normalized manipulability
                if normalized_manip > 0.6:
                    color = (100, 255, 140)  # Green (good)
                elif normalized_manip > 0.3:
                    color = (255, 220, 100)  # Yellow (medium)
                else:
                    color = (255, 150, 150)  # Red (low)
            else:
                color = COLOR_TEXT_SECONDARY
            text_surf = font.render(line, True, color)
            screen.blit(text_surf, (hud_x + hud_padding, hud_y + hud_padding + i * 20))
        hud_y += manip_bg.height + 8
        
        # Controls panel
        control_lines = [
            "Controls:",
            "A/Z, S/X, D/C: Adjust joints ±3°",
            "U/M: Flip Configuration",
            "R: Reset  +/-: Scale links",
            "3: Toggle Workspace  Click: Set IK target",
            "J: Joint input  T: Target input  L: Link lengths",
        ]
        max_width = max([font.size(line)[0] for line in control_lines]) + hud_padding * 2
        control_bg = pygame.Rect(hud_x, hud_y, max_width, len(control_lines) * 20 + hud_padding * 2)
        pygame.draw.rect(screen, COLOR_PANEL, control_bg)
        pygame.draw.rect(screen, COLOR_PANEL_BORDER, control_bg, 2)
        for i, line in enumerate(control_lines):
            color = COLOR_TEXT_PRIMARY if i == 0 else COLOR_TEXT_SECONDARY
            text_surf = font.render(line, True, color)
            screen.blit(text_surf, (hud_x + hud_padding, hud_y + hud_padding + i * 20))

        # Draw input overlay last
        input_ui.draw(screen)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()


