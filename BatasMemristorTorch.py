import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class BatasMemristorTorch(nn.Module):
    def __init__(self, RON, ROFF, D, t0, v0):
        super(BatasMemristorTorch, self).__init__()
        self.RON = nn.Parameter(torch.tensor(RON, dtype=torch.float64))
        self.ROFF = nn.Parameter(torch.tensor(ROFF, dtype=torch.float64))
        self.D = nn.Parameter(torch.tensor(D, dtype=torch.float64))
        self.t0 = torch.tensor(t0, dtype=torch.float64)
        self.v0 = torch.tensor(v0, dtype=torch.float64)
        self.w = self.calculate_initial_width()
        self.uv = torch.tensor(1e-14, dtype=torch.float64)  # Example mobility value

    def calculate_initial_width(self):
        # Calculate initial width based on provided initial voltage and time
        w0 = self.v0 * self.t0 / self.RON
        return nn.Parameter(w0)

    def ResetInitVals(self, InitVals):
        # Reset memristor state to new initial values
        RON, ROFF, D, t0, v0 = InitVals
        self.RON.data = torch.tensor(RON, dtype=torch.float64)
        self.ROFF.data = torch.tensor(ROFF, dtype=torch.float64)
        self.D.data = torch.tensor(D, dtype=torch.float64)
        self.t0 = torch.tensor(t0, dtype=torch.float64)
        self.v0 = torch.tensor(v0, dtype=torch.float64)
        self.w.data = self.calculate_initial_width()

    def UpdateVals(self, Vin):
        # Calculate change in w based on applied voltage Vin
        dw = Vin * self.uv * self.RON / self.D
        self.w.data += dw

    def CalculateInitialResistance(self):
        # Calculate initial resistance based on initial width
        r = self.RON * self.w / self.D + self.ROFF * (1 - self.w / self.D)
        return r

    def GetInitVals(self, InitStates):
        # Get initial resistance based on initial width
        return self.CalculateInitialResistance().item()

    def GetVals(self, VinVals, dt):
        # Reset memristor state to initial values
        self.w.data = self.calculate_initial_width()

        # Calculate resistance for each applied voltage in VinVals
        resistance_values = []
        for Vin in VinVals:
            # Update memristor state based on applied voltage
            self.UpdateVals(Vin)
            # Calculate resistance and append to the list
            resistance = self.CalculateInitialResistance()
            resistance_values.append(resistance.item())
        return resistance_values

    def CalculateCurrent(self, VinVals):
        # Calculate current for each applied voltage in VinVals
        current_values = []
        for Vin in VinVals:
            # Calculate resistance for the applied voltage
            resistance = self.RON * self.w / self.D + self.ROFF * (1 - self.w / self.D)
            # Calculate current using Ohm's Law
            current = Vin / resistance
            current_values.append(current.item())
        return current_values

def generate_waveform(wave_type, amplitude, frequency, duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    if wave_type == 'sine':
        return amplitude * np.sin(2 * np.pi * frequency * t)
    elif wave_type == 'square':
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    elif wave_type == 'triangle':
        return amplitude * 2 * np.abs(np.arcsin(np.sin(2 * np.pi * frequency * t)) / np.pi)
    else:
        raise ValueError("Unsupported wave type. Choose 'sine', 'square', or 'triangle'.")

# Define the initial values
RON = 100  # ON resistance (Ω)
ROFF = 160 * RON  # OFF resistance (Ω)
D = 10  # Maximum drift distance (nm)
t0 = 10  # Minimum drift time (ms)
v0 = 1  # Initial voltage (V)

# Create memristor instance
memristor = BatasMemristorTorch(RON=RON, ROFF=ROFF, D=D, t0=t0, v0=v0)

# Generate a sine waveform for input voltage
wave_type = 'sine'  # Choose 'sine', 'square', or 'triangle'
amplitude = 1.0  # Voltage amplitude
frequency = 1.0  # Frequency in Hz
duration = 2.0  # Duration in seconds
sampling_rate = 1000  # Sampling rate in Hz

Vin_values = generate_waveform(wave_type, amplitude, frequency, duration, sampling_rate)
Vin_values = torch.tensor(Vin_values, dtype=torch.float64)
dt = 1 / sampling_rate  # Timestep

# Calculate current values for the applied voltages
current_values = memristor.CalculateCurrent(Vin_values)

# Print current values
print("Current values:", current_values)

# Plot hysteresis response: I vs. V
plt.plot(Vin_values.numpy(), current_values, marker='o')
plt.xlabel('Applied Voltage (Vin)')
plt.ylabel('Current (I)')
plt.title('Hysteresis Response: I vs. V')
plt.grid(True)
plt.show()

class MNA:
    def __init__(self, memristors, dt=0.001, t_end=2.0):
        self.memristors = memristors
        self.dt = dt
        self.t_end = t_end
        self.time_steps = int(t_end / dt)
        self.vs = 2 * np.sin(2 * np.pi * 1 * np.arange(0, t_end, dt))  # 2V amplitude, 1Hz frequency sine wave

    def construct_matrix(self, t):
        # Construct matrix A and vector Z at time t
        R1, R2, R3, R4, R5 = [m.CalculateResistance().item() for m in self.memristors]
        vs_t = self.vs[t]

        A = np.array([
            [1, -1, 0, 0, 0, 0],
            [-1, 1 + 1/R1 + 1/R4, -1/R1, 0, -1/R4, 0],
            [0, -1/R1, 1 + 1/R1 + 1/R2, -1/R2, 0, 0],
            [0, 0, -1/R2, 1 + 1/R2 + 1/R3 + 1/R5, -1/R3, -1/R5],
            [0, -1/R4, 0, -1/R3, 1/R4 + 1/R3 + 1/R5, -1/R5],
            [0, 0, 0, 0, -1, 1]
        ])

        Z = np.array([vs_t, 0, 0, 0, 0, 0])
        
        return A, Z

    def solve(self):
        voltages = np.zeros((self.time_steps, 4))  # Nodes 1, 2, 3, and source

        for t in range(self.time_steps):
            A, Z = self.construct_matrix(t)
            X = np.linalg.solve(A, Z)  # Solve for voltages
            voltages[t, :] = X[1:5]  # Store node voltages
            
            # Update memristor resistances
            for i, mem in enumerate(self.memristors):
                mem.UpdateVals(torch.tensor(voltages[t, i % 3]))  # Update based on node voltages

        return voltages

# Define the initial values for memristors
RON = 100  # ON resistance (Ω)
ROFF = 160 * RON  # OFF resistance (Ω)
D = 10  # Overall device length
t0 = 10  # Minimum drift time (ms)
v0 = 1  # Initial voltage (V)

# Create memristor instances
memristors = [BatasMemristorTorch(RON, ROFF, D, t0, v0) for _ in range(5)]

# Create MNA instance and solve for node voltages
mna = MNA(memristors)
voltages = mna.solve()

# Plot voltage at node 3
time = np.arange(0, mna.t_end, mna.dt)
plt.plot(time, voltages[:, 2], label='$v_3(t)$')
plt.xlabel('time (s)')
plt.ylabel('Voltage (V)')
plt.title('Voltage at Node 3 of a Memristive Circuit')
plt.legend()
plt.grid(True)
plt.show()
