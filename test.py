import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class BatasMemristorTorch(nn.Module):
    def __init__(self, RON, ROFF, D, t0, v0):
        super(BatasMemristorTorch, self).__init__()
        self.RON = nn.Parameter(torch.tensor(RON, dtype=torch.float64))
        self.ROFF = nn.Parameter(torch.tensor(ROFF, dtype=torch.float64))
        self.D = nn.Parameter(torch.tensor(D, dtype=torch.float64))
        self.t0 = t0
        self.v0 = v0
        self.w = self.calculate_initial_width()

    def calculate_initial_width(self):
        # Calculate initial width based on provided initial voltage and time
        w0 = self.v0 * self.t0 / self.RON
        return nn.Parameter(torch.tensor(w0, dtype=torch.float64))

    def ResetInitVals(self, RON, ROFF, D, t0, v0):
        # Reset memristor state to new initial values
        self.RON.data = torch.tensor(RON, dtype=torch.float64)
        self.ROFF.data = torch.tensor(ROFF, dtype=torch.float64)
        self.D.data = torch.tensor(D, dtype=torch.float64)
        self.t0 = t0
        self.v0 = v0
        self.w.data = self.calculate_initial_width()

    def UpdateVals(self, Vin):
        # Calculate change in w based on applied voltage Vin
        dw = Vin * self.RON / self.D
        self.w.data += dw

    def CalculateInitialResistance(self):
        # Calculate initial resistance based on initial width
        r = self.RON * self.w / self.D + self.ROFF * (1 - self.w / self.D)
        return r

    def GetInitialVals(self):
        # Get initial resistance based on initial width
        return self.CalculateInitialResistance()

    def GetVals(self, VinVals):
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

# Define the initial values
RON = 100  # ON resistance (Ω)
ROFF = 160 * RON  # OFF resistance (Ω)
D = 10  # Maximum drift distance (nm)
t0 = 10  # Minimum drift time (ms)
v0 = 1  # Initial voltage (V)

# Create memristor instance
memristor = BatasMemristorTorch(RON=RON, ROFF=ROFF, D=D, t0=t0, v0=v0)

# Define voltage inputs
Vin_values = [1.0, 0.5, 0.8]

# Get resistance values for the applied voltages
resistance_values = memristor.GetVals(Vin_values)

# Print resistance values
print("Resistance values:", resistance_values)

# Plot resistance values over time
plt.plot(Vin_values, resistance_values)
plt.xlabel('Applied Voltage (Vin)')
plt.ylabel('Resistance')
plt.title('Resistance vs. Applied Voltage')
plt.show()
