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

    def CalculateResistance(self):
        # Calculate resistance based on width
        r = self.RON * self.w / self.D + self.ROFF * (1 - self.w / self.D)
        return r

    def GetInitVals(self, InitStates):
        # Get initial resistance based on initial width
        return self.CalculateResistance().item()

    def GetVals(self, VinVals, dt):
        # Reset memristor state to initial values
        self.w.data = self.calculate_initial_width()

        # Calculate resistance for each applied voltage in VinVals
        resistance_values = []
        for Vin in VinVals:
            # Update memristor state based on applied voltage
            self.UpdateVals(Vin)
            # Calculate resistance and append to the list
            resistance = self.CalculateResistance()
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

class MNA:
    def __init__(self, memristors, dt=0.001, t_end=2.0):
        self.memristors = memristors
        self.dt = dt
        self.t_end = t_end
        self.time_steps = int(t_end / dt)
        self.vs = 2 * np.sin(2 * np.pi * 1 * np.arange(0, t_end, dt))  # 2V amplitude, 1Hz frequency sine wave

    def construct_matrix(self, t):
        # Construct matrix A and vector Z at time t
        resistances = [m.CalculateResistance().item() for m in self.memristors]
        vs_t = self.vs[t]

        # Example extended MNA matrix and vector for a larger network
        # Note: This should be adapted to the specific network structure
        size = len(self.memristors) + 1
        A = np.zeros((size, size))
        Z = np.zeros(size)

        # Populate matrix A and vector Z with values based on the circuit
        A[0, 0] = 1
        Z[0] = vs_t

        for i in range(1, size):
            A[i, i] = 1 + sum(1 / r for r in resistances)
            A[i, 0] = -1 / resistances[i - 1]
            A[0, i] = -1 / resistances[i - 1]

        return A, Z

    def solve(self):
        voltages = np.zeros((self.time_steps, len(self.memristors) + 1))  # Nodes and source

        for t in range(self.time_steps):
            A, Z = self.construct_matrix(t)
            X = np.linalg.solve(A, Z)  # Solve for voltages
            voltages[t, :] = X  # Store node voltages
            
            # Update memristor resistances
            for i, mem in enumerate(self.memristors):
                mem.UpdateVals(torch.tensor(voltages[t, i + 1]))  # Update based on node voltages

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
plt.plot(time, voltages[:, 3], label='$v_3(t)$')
plt.xlabel('time (s)')
plt.ylabel('Voltage (V)')
plt.title('Voltage at Node 3 of a Memristive Circuit')
plt.legend()
plt.grid(True)
plt.show()
    