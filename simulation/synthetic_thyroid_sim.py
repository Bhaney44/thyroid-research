"""
Synthetic Thyroid Implant Simulation
=====================================
Models a subcutaneous T4-delivery implant releasing a configurable dose
of levothyroxine (T4) every 12 hours, and tracks the resulting Free T4
serum concentration over time.

Pharmacokinetic model
---------------------
Two-compartment, first-order kinetics:

  Absorbing pool (A):  dA/dt = -ka * A        ka  = ln2 / t_half_absorb
  Free T4 pool  (F):  dF/dt = ka * A * scale - ke * F    ke  = ln2 / t_half_elim

Where:
  t_half_absorb  ~ 1.2 h   (subcutaneous absorption)
  t_half_elim    ~ 6.5 days → simplified to ~6.5 h for demo speed
  scale          = DOSE_TO_NGDL conversion factor

Normal Free T4 range: 0.8 – 1.8 ng/dL

Usage
-----
Run directly:
    python synthetic_thyroid_sim.py

Adjust constants at the top of the file to explore different regimens.

Requirements
------------
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from matplotlib.patches import FancyArrowPatch
import matplotlib.animation as animation

# ─────────────────────────────────────────────
#  ADJUSTABLE PARAMETERS  (edit these freely)
# ─────────────────────────────────────────────
DOSE_MCG          = 100.0    # mcg of T4 per dose
DOSE_INTERVAL_H   = 12.0     # hours between doses
SIM_DURATION_H    = 240.0    # total simulation time (hours) — 10 days
DT                = 0.05     # integration step (hours)

# Pharmacokinetics
T_HALF_ABSORB_H   = 1.2      # absorption half-life (hours)
T_HALF_ELIM_H     = 6.5      # elimination half-life (hours) — compressed for demo
                              # real T4 half-life is ~6-7 days; use 156 for realism
DOSE_TO_NGDL      = 0.012    # mcg → ng/dL conversion per unit absorbed

INITIAL_FREE_T4   = 0.4      # starting serum Free T4 (ng/dL) — hypothyroid baseline

# Reference range
NORMAL_LOW        = 0.8      # ng/dL
NORMAL_HIGH       = 1.8      # ng/dL

# Animation
REAL_MS_PER_STEP  = 20       # ms per animation frame (lower = faster)
SIM_STEPS_PER_FRAME = 3      # simulation steps per animation frame
# ─────────────────────────────────────────────


def run_full_simulation(dose_mcg, dose_interval_h, duration_h, dt,
                        t_half_absorb, t_half_elim, dose_to_ngdl, init_t4):
    """Run entire simulation ahead of time. Returns arrays for plotting."""
    ka = np.log(2) / t_half_absorb
    ke = np.log(2) / t_half_elim

    n_steps   = int(duration_h / dt)
    times     = np.zeros(n_steps)
    free_t4   = np.zeros(n_steps)
    absorbing = 0.0
    ft4       = init_t4
    next_dose = 0.0
    dose_times = []

    for i in range(n_steps):
        t = i * dt
        if t >= next_dose:
            absorbing += dose_mcg
            next_dose += dose_interval_h
            dose_times.append(t)

        dA = -ka * absorbing
        dF = ka * absorbing * dose_to_ngdl - ke * ft4
        absorbing += dA * dt
        absorbing  = max(0.0, absorbing)
        ft4       += dF * dt
        ft4        = max(0.0, ft4)

        times[i]   = t
        free_t4[i] = ft4

    return times, free_t4, dose_times


class SyntheticThyroidSim:
    def __init__(self):
        # Simulation state
        self.dose_mcg       = DOSE_MCG
        self.dose_interval  = DOSE_INTERVAL_H
        self.t_half_absorb  = T_HALF_ABSORB_H
        self.t_half_elim    = T_HALF_ELIM_H

        self.reset_state()

        # ── Figure layout ──────────────────────────────────────────────
        self.fig = plt.figure(figsize=(14, 9), facecolor='#0e1117')
        self.fig.canvas.manager.set_window_title('Synthetic Thyroid Implant Simulation')

        gs = gridspec.GridSpec(
            3, 2,
            figure=self.fig,
            left=0.07, right=0.96,
            top=0.93, bottom=0.18,
            hspace=0.45, wspace=0.3
        )

        self.ax_chart  = self.fig.add_subplot(gs[0:2, :])   # main T4 chart (spans both cols, 2 rows)
        self.ax_device = self.fig.add_subplot(gs[2, 0])     # device diagram
        self.ax_stats  = self.fig.add_subplot(gs[2, 1])     # live stats

        for ax in [self.ax_chart, self.ax_device, self.ax_stats]:
            ax.set_facecolor('#161b22')
            for spine in ax.spines.values():
                spine.set_color('#30363d')

        # ── Sliders & buttons ─────────────────────────────────────────
        slider_color = '#21262d'

        ax_dose   = self.fig.add_axes([0.07, 0.11, 0.25, 0.025])
        ax_interv = self.fig.add_axes([0.07, 0.07, 0.25, 0.025])
        ax_speed  = self.fig.add_axes([0.07, 0.03, 0.25, 0.025])

        self.sl_dose   = Slider(ax_dose,   'Dose (mcg)',    25,  200, valinit=DOSE_MCG,        valstep=25,  color='#1f6feb')
        self.sl_interv = Slider(ax_interv, 'Interval (h)', 4,   24,  valinit=DOSE_INTERVAL_H, valstep=2,   color='#1f6feb')
        self.sl_speed  = Slider(ax_speed,  'Speed',        1,   10,  valinit=SIM_STEPS_PER_FRAME, valstep=1, color='#388bfd')

        for sl in [self.sl_dose, self.sl_interv, self.sl_speed]:
            sl.label.set_color('#8b949e')
            sl.valtext.set_color('#c9d1d9')
            sl.ax.set_facecolor(slider_color)

        self.sl_dose.on_changed(self._on_param_change)
        self.sl_interv.on_changed(self._on_param_change)

        ax_btn_reset  = self.fig.add_axes([0.40, 0.03, 0.08, 0.055])
        ax_btn_pause  = self.fig.add_axes([0.50, 0.03, 0.08, 0.055])
        ax_btn_full   = self.fig.add_axes([0.60, 0.03, 0.12, 0.055])

        self.btn_reset = Button(ax_btn_reset, 'Reset',     color='#21262d', hovercolor='#30363d')
        self.btn_pause = Button(ax_btn_pause, 'Pause',     color='#21262d', hovercolor='#30363d')
        self.btn_full  = Button(ax_btn_full,  'Full run ▶', color='#1f6feb', hovercolor='#388bfd')

        for btn in [self.btn_reset, self.btn_pause, self.btn_full]:
            btn.label.set_color('#c9d1d9')

        self.btn_reset.on_clicked(self._on_reset)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_full.on_clicked(self._on_full_run)

        # ── Title ──────────────────────────────────────────────────────
        self.fig.text(0.5, 0.965, 'Synthetic Thyroid Implant — T4 Delivery Simulation',
                      ha='center', va='top', color='#c9d1d9', fontsize=13, fontweight='bold')

        # ── Chart setup ────────────────────────────────────────────────
        self._setup_chart()
        self._setup_device_panel()
        self._setup_stats_panel()

        # ── Animation ──────────────────────────────────────────────────
        self.paused   = False
        self.particle_frame = 0
        self.dose_flash = 0   # frames to show dose flash

        self.anim = animation.FuncAnimation(
            self.fig, self._animate,
            interval=REAL_MS_PER_STEP,
            blit=False, cache_frame_data=False
        )

    # ─────────────────────────────────────────
    #  State management
    # ─────────────────────────────────────────
    def reset_state(self):
        self.ka = np.log(2) / self.t_half_absorb
        self.ke = np.log(2) / self.t_half_elim
        self.t          = 0.0
        self.free_t4    = INITIAL_FREE_T4
        self.absorbing  = 0.0
        self.next_dose  = 0.0
        self.dose_count = 0
        self.times_hist = []
        self.t4_hist    = []
        self.dose_times = []
        self.finished   = False

    # ─────────────────────────────────────────
    #  Chart
    # ─────────────────────────────────────────
    def _setup_chart(self):
        ax = self.ax_chart
        ax.set_xlim(0, min(SIM_DURATION_H, 48))
        ax.set_ylim(0, 2.6)
        ax.set_xlabel('Time (hours)', color='#8b949e', fontsize=9)
        ax.set_ylabel('Free T4 (ng/dL)', color='#8b949e', fontsize=9)
        ax.tick_params(colors='#8b949e', labelsize=8)

        # Reference band
        ax.axhspan(NORMAL_LOW, NORMAL_HIGH, alpha=0.12, color='#1f6feb', label='Normal range')

        # Reference lines
        ax.axhline(NORMAL_HIGH, color='#f85149', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.axhline(NORMAL_LOW,  color='#388bfd', linewidth=0.8, linestyle='--', alpha=0.7)

        ax.text(0.5, NORMAL_HIGH + 0.04, 'Upper normal (1.8 ng/dL)',
                color='#f85149', fontsize=7.5, transform=ax.get_yaxis_transform(), alpha=0.85)
        ax.text(0.5, NORMAL_LOW  + 0.04, 'Lower normal (0.8 ng/dL)',
                color='#388bfd', fontsize=7.5, transform=ax.get_yaxis_transform(), alpha=0.85)

        self.line_t4, = ax.plot([], [], color='#3fb950', linewidth=1.8, label='Free T4', zorder=5)
        self.vline    = ax.axvline(x=0, color='#8b949e', linewidth=0.8, alpha=0.5)

        # Dose markers container (will add vlines dynamically)
        self.dose_vlines = []

        ax.legend(loc='upper right', fontsize=8, facecolor='#21262d',
                  edgecolor='#30363d', labelcolor='#c9d1d9')
        ax.set_title('Free T4 serum concentration', color='#8b949e', fontsize=9, pad=6)

    def _setup_device_panel(self):
        ax = self.ax_device
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Implant schematic', color='#8b949e', fontsize=9, pad=4)

        # Device housing
        housing = mpatches.FancyBboxPatch((0.5, 1.5), 9, 3,
            boxstyle='round,pad=0.15', linewidth=1,
            edgecolor='#388bfd', facecolor='#161b22')
        ax.add_patch(housing)

        # Reservoir
        res = mpatches.FancyBboxPatch((0.8, 1.8), 3, 2.4,
            boxstyle='round,pad=0.1', linewidth=0.8,
            edgecolor='#1f6feb', facecolor='#0d1117')
        ax.add_patch(res)
        ax.text(2.3, 3.2, 'T4', color='#58a6ff', fontsize=11, fontweight='bold', ha='center')
        ax.text(2.3, 2.65, 'reservoir', color='#8b949e', fontsize=7, ha='center')
        self.res_fill = mpatches.FancyBboxPatch((0.85, 1.85), 2.9, 0.0,
            boxstyle='round,pad=0.05', linewidth=0,
            edgecolor='none', facecolor='#1f6feb', alpha=0.4)
        ax.add_patch(self.res_fill)

        # Pump
        pump = mpatches.FancyBboxPatch((4.2, 1.8), 2.2, 2.4,
            boxstyle='round,pad=0.1', linewidth=0.8,
            edgecolor='#7c3aed', facecolor='#0d1117')
        ax.add_patch(pump)
        ax.text(5.3, 3.1, 'Micro', color='#a78bfa', fontsize=7.5, ha='center')
        ax.text(5.3, 2.7, 'pump', color='#a78bfa', fontsize=7.5, ha='center')

        # Release port
        port = mpatches.FancyBboxPatch((6.8, 2.1), 1.0, 1.8,
            boxstyle='round,pad=0.08', linewidth=0.8,
            edgecolor='#3fb950', facecolor='#0d1117')
        ax.add_patch(port)
        ax.text(7.3, 3.1, 'Port', color='#3fb950', fontsize=7, ha='center')

        # Blood vessel outline
        vessel = mpatches.FancyBboxPatch((8.2, 1.6), 1.3, 2.8,
            boxstyle='round,pad=0.1', linewidth=1.5,
            edgecolor='#f85149', facecolor='#1a0a0a', alpha=0.7)
        ax.add_patch(vessel)
        ax.text(8.85, 3.6, '♥', color='#f85149', fontsize=9, ha='center')
        ax.text(8.85, 2.0, 'blood', color='#f85149', fontsize=6.5, ha='center')

        # Flow arrows (static)
        ax.annotate('', xy=(4.15, 3.0), xytext=(3.75, 3.0),
                    arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=1.0))
        ax.annotate('', xy=(6.75, 3.0), xytext=(6.45, 3.0),
                    arrowprops=dict(arrowstyle='->', color='#a78bfa', lw=1.0))
        ax.annotate('', xy=(8.15, 3.0), xytext=(7.85, 3.0),
                    arrowprops=dict(arrowstyle='->', color='#3fb950', lw=1.2))

        # Particle dots (animated)
        self.particles = []
        for _ in range(6):
            dot, = ax.plot([], [], 'o', color='#3fb950', markersize=4, alpha=0, zorder=10)
            self.particles.append(dot)

        # Dose flash rect
        self.dose_flash_rect = mpatches.FancyBboxPatch((0.5, 1.5), 9, 3,
            boxstyle='round,pad=0.15', linewidth=2,
            edgecolor='#3fb950', facecolor='none', alpha=0, zorder=20)
        ax.add_patch(self.dose_flash_rect)

        # Timer bar
        ax.text(0.8, 1.25, 'Next dose:', color='#8b949e', fontsize=7)
        self.timer_bg = mpatches.Rectangle((2.5, 1.1), 7, 0.35,
            linewidth=0.5, edgecolor='#30363d', facecolor='#21262d')
        ax.add_patch(self.timer_bg)
        self.timer_bar = mpatches.Rectangle((2.5, 1.1), 0, 0.35,
            linewidth=0, facecolor='#3fb950')
        ax.add_patch(self.timer_bar)

        ax.text(5, 0.6, 'Time labels', color='#30363d', fontsize=7, ha='center')
        self.dose_label = ax.text(5, 0.55, '', color='#8b949e', fontsize=7, ha='center')

    def _setup_stats_panel(self):
        ax = self.ax_stats
        ax.axis('off')
        ax.set_title('Live stats', color='#8b949e', fontsize=9, pad=4)
        c = '#8b949e'
        self.stat_texts = {}
        labels = [
            ('free_t4',    'Free T4:',       1.0),
            ('status',     'Status:',         0.82),
            ('dose_count', 'Doses given:',    0.64),
            ('total_mcg',  'Total delivered:',0.46),
            ('sim_time',   'Sim time:',        0.28),
            ('next_dose',  'Next dose in:',    0.10),
        ]
        for key, label, y in labels:
            ax.text(0.02, y, label, color=c, fontsize=9,
                    transform=ax.transAxes, va='center')
            self.stat_texts[key] = ax.text(0.55, y, '—', color='#c9d1d9',
                                           fontsize=9, fontweight='bold',
                                           transform=ax.transAxes, va='center')

    # ─────────────────────────────────────────
    #  Simulation step
    # ─────────────────────────────────────────
    def _step(self, dt):
        dosed = False
        if self.t >= self.next_dose:
            self.absorbing  += self.dose_mcg
            self.next_dose  += self.dose_interval
            self.dose_count += 1
            self.dose_times.append(self.t)
            dosed = True

        dA = -self.ka * self.absorbing
        dF = self.ka * self.absorbing * DOSE_TO_NGDL - self.ke * self.free_t4
        self.absorbing = max(0.0, self.absorbing + dA * dt)
        self.free_t4   = max(0.0, self.free_t4   + dF * dt)
        self.t        += dt

        self.times_hist.append(self.t)
        self.t4_hist.append(self.free_t4)

        if self.t >= SIM_DURATION_H:
            self.finished = True

        return dosed

    # ─────────────────────────────────────────
    #  Animation callback
    # ─────────────────────────────────────────
    def _animate(self, frame):
        if self.paused or self.finished:
            return

        steps = max(1, int(self.sl_speed.val))
        dosed_this_frame = False

        for _ in range(steps):
            if not self.finished:
                dosed = self._step(DT)
                if dosed:
                    dosed_this_frame = True

        # Update main chart
        if self.times_hist:
            t_arr  = np.array(self.times_hist)
            t4_arr = np.array(self.t4_hist)
            self.line_t4.set_data(t_arr, t4_arr)

            # Scroll x-axis to follow current time
            window = 48
            x_max = max(self.t, window)
            self.ax_chart.set_xlim(max(0, self.t - window * 0.85), x_max)
            self.vline.set_xdata([self.t, self.t])

            # Dose markers (add only new ones)
            while len(self.dose_vlines) < len(self.dose_times):
                dt_x = self.dose_times[len(self.dose_vlines)]
                vl = self.ax_chart.axvline(dt_x, color='#a78bfa',
                                           linewidth=0.8, alpha=0.5, linestyle=':')
                self.dose_vlines.append(vl)

        # Device panel updates
        if dosed_this_frame:
            self.dose_flash = 12
            self.particle_frame = 0

        # Dose flash
        if self.dose_flash > 0:
            self.dose_flash_rect.set_alpha(0.6 * (self.dose_flash / 12))
            self.dose_flash -= 1
        else:
            self.dose_flash_rect.set_alpha(0)

        # Particle animation
        px_positions = [7.85, 8.0, 7.9, 8.05, 7.95, 8.1]
        py_positions = [2.5, 2.8, 3.0, 3.2, 3.4, 3.6]
        if self.dose_flash > 0:
            pf = self.particle_frame
            for i, dot in enumerate(self.particles):
                offset = i * 2
                age = pf - offset
                if 0 <= age <= 10:
                    alpha = max(0, 1 - age / 10)
                    dx = age * 0.04
                    dot.set_data([px_positions[i] + dx], [py_positions[i]])
                    dot.set_alpha(alpha)
                else:
                    dot.set_alpha(0)
            self.particle_frame += 1
        else:
            for dot in self.particles:
                dot.set_alpha(0)

        # Timer bar
        time_since = self.t - (self.next_dose - self.dose_interval)
        frac = min(1.0, time_since / self.dose_interval)
        self.timer_bar.set_width(7 * frac)
        time_to_next = max(0, self.next_dose - self.t)
        self.dose_label.set_text(f'Next dose in {time_to_next:.1f}h')

        # Reservoir fill (cosmetic depletion)
        fill_frac = max(0.05, 1.0 - self.dose_count * 0.015)
        self.res_fill.set_height(2.3 * fill_frac)

        # Stats panel
        t4 = self.free_t4
        self.stat_texts['free_t4'].set_text(f'{t4:.3f} ng/dL')

        if t4 < NORMAL_LOW:
            status, col = 'Below normal ▼', '#f0a132'
        elif t4 > NORMAL_HIGH:
            status, col = 'Above normal ▲', '#f85149'
        else:
            status, col = 'Normal range ✓', '#3fb950'
        self.stat_texts['status'].set_text(status)
        self.stat_texts['status'].set_color(col)

        self.stat_texts['dose_count'].set_text(str(self.dose_count))
        self.stat_texts['total_mcg'].set_text(f'{self.dose_count * self.dose_mcg:.0f} mcg')
        day  = int(self.t // 24) + 1
        hour = self.t % 24
        self.stat_texts['sim_time'].set_text(f'Day {day}, {hour:.1f}h')
        self.stat_texts['next_dose'].set_text(f'{time_to_next:.1f}h')

        if self.finished:
            self.ax_chart.set_title('Free T4 serum concentration  [simulation complete]',
                                    color='#3fb950', fontsize=9, pad=6)

    # ─────────────────────────────────────────
    #  Widget callbacks
    # ─────────────────────────────────────────
    def _on_param_change(self, val):
        self.dose_mcg      = self.sl_dose.val
        self.dose_interval = self.sl_interv.val
        self._on_reset(None)

    def _on_reset(self, event):
        self.dose_mcg      = self.sl_dose.val
        self.dose_interval = self.sl_interv.val
        self.reset_state()
        self.line_t4.set_data([], [])
        self.vline.set_xdata([0, 0])
        for vl in self.dose_vlines:
            vl.remove()
        self.dose_vlines.clear()
        self.ax_chart.set_xlim(0, 48)
        self.ax_chart.set_title('Free T4 serum concentration', color='#8b949e', fontsize=9, pad=6)
        self.paused = False
        self.btn_pause.label.set_text('Pause')
        for dot in self.particles:
            dot.set_alpha(0)
        self.dose_flash_rect.set_alpha(0)

    def _on_pause(self, event):
        self.paused = not self.paused
        self.btn_pause.label.set_text('Resume' if self.paused else 'Pause')

    def _on_full_run(self, event):
        """Pre-compute and display the full simulation instantly."""
        times, t4_vals, dose_ts = run_full_simulation(
            self.dose_mcg, self.dose_interval, SIM_DURATION_H, DT,
            self.t_half_absorb, self.t_half_elim, DOSE_TO_NGDL, INITIAL_FREE_T4
        )

        # Clear existing animated state
        for vl in self.dose_vlines:
            vl.remove()
        self.dose_vlines.clear()

        self.line_t4.set_data(times, t4_vals)
        self.ax_chart.set_xlim(0, SIM_DURATION_H)
        self.vline.set_xdata([SIM_DURATION_H, SIM_DURATION_H])

        for dt_x in dose_ts:
            vl = self.ax_chart.axvline(dt_x, color='#a78bfa',
                                       linewidth=0.6, alpha=0.4, linestyle=':')
            self.dose_vlines.append(vl)

        self.ax_chart.set_title(
            f'Free T4 — full {SIM_DURATION_H/24:.0f}-day simulation  '
            f'(Dose: {self.dose_mcg:.0f} mcg, Interval: {self.dose_interval:.0f}h)',
            color='#3fb950', fontsize=9, pad=6
        )

        # Update state to end of sim
        self.t          = times[-1]
        self.free_t4    = t4_vals[-1]
        self.times_hist = list(times)
        self.t4_hist    = list(t4_vals)
        self.dose_times = dose_ts
        self.dose_count = len(dose_ts)
        self.finished   = True
        self.paused     = True

        # Stats
        t4 = self.free_t4
        self.stat_texts['free_t4'].set_text(f'{t4:.3f} ng/dL')
        self.stat_texts['dose_count'].set_text(str(self.dose_count))
        self.stat_texts['total_mcg'].set_text(f'{self.dose_count * self.dose_mcg:.0f} mcg')
        self.stat_texts['sim_time'].set_text(f'Day {SIM_DURATION_H/24:.0f}, done')
        self.stat_texts['next_dose'].set_text('—')

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("Synthetic Thyroid Implant Simulation")
    print(f"  Dose: {DOSE_MCG} mcg every {DOSE_INTERVAL_H}h")
    print(f"  PK: absorption t½={T_HALF_ABSORB_H}h, elimination t½={T_HALF_ELIM_H}h")
    print(f"  Normal Free T4: {NORMAL_LOW}–{NORMAL_HIGH} ng/dL")
    print("\nControls:")
    print("  Dose / Interval sliders  — change regimen (resets sim)")
    print("  Speed slider             — animation speed multiplier")
    print("  Pause / Resume           — toggle animation")
    print("  Reset                    — restart from t=0")
    print("  Full run ▶               — compute & display all 10 days instantly\n")

    sim = SyntheticThyroidSim()
    sim.show()
