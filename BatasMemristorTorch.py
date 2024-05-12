import numpy as np
import torch, os
import torch.nn as nn

class BatasMemristorTorch(nn.Module):
    def __init__(self, InitVals=0.0, Percent=0.0, Size=None, DecayEffect=False, \
            Theta=1.0, Vth=0.0, Verbose=False):

        # set the function name
        FunctionName = "Oblea::__init__()"

        # perecent
        self.Percent = Percent

        # memristor model name
        self.ModelName = "Oblea"

        # save parameters
        self.DecayEffect = DecayEffect

        # **********************************************************************
        # flags
        # **********************************************************************
        if self.Percent > 0.0:
            self.Ideal = False
        else:
            self.Ideal = True

        self.Verbose = Verbose

        # **********************************************************************
        # constants for the model
        # **********************************************************************
        self.Vp = self.VpNom = torch.tensor(0.16, dtype=torch.float64)
        self.Vn = self.VnNom = torch.tensor(0.15, dtype=torch.float64)
        self.Ap = self.ApNom = torch.tensor(4000, dtype=torch.float64)
        self.An = self.AnNom = torch.tensor(4000, dtype=torch.float64)
        self.Xp = self.XpNom = torch.tensor(0.3, dtype=torch.float64)
        self.Xn = self.XnNom = torch.tensor(0.5, dtype=torch.float64)
        self.AlphaP = self.AlphaPNom = torch.tensor(1.0, dtype=torch.float64)
        self.AlphaN = self.AlphaNNom = torch.tensor(5.0, dtype=torch.float64)
        self.a1 = self.a1Nom = torch.tensor(0.17, dtype=torch.float64)
        self.a2 = self.a2Nom = torch.tensor(0.17, dtype=torch.float64)
        self.b  = self.bNom  = torch.tensor(0.05, dtype=torch.float64)
        self.Xo = self.XoNom = torch.tensor(0.11, dtype=torch.float64)
        self.Eta = self.EtaNom = torch.tensor(1.0, dtype=torch.float64)

        # **********************************************************************
        # check value for x
        # **********************************************************************
        self.CheckX = 1.0 - self.Xn

        # **********************************************************************
        # these values are specific to the device
        # **********************************************************************
        self.RhoMin = torch.tensor(0.01, dtype=torch.float64)
        self.RhoMax = torch.tensor(0.9983, dtype=torch.float64)
        self.DeltaRho = self.RhoMax - self.RhoMin
        self.VRead  = torch.tensor(0.1, dtype=torch.float64)

        # **********************************************************************
        # the threshold voltage for crossbar classifier; it is not needed for
        # reservoir
        # **********************************************************************
        self.Vth = (torch.add(self.Vp, self.Vn) / 2.0).item()

        # print("self.Vth = ", self.Vth)
        # exit()

        # **********************************************************************
        # get min and max values for conductance and resistance
        # **********************************************************************
        self.Gmin, self.Gmax, self.Rmin, self.Rmax = self._CalMinMaxValues()

        # state variable
        self.Rho = self._CalRho(InitVals)

        # set the inputs and outputs
        if Size is not None:
            (self.Inputs, self.Outputs) = Size

        # set the internal state variables
        self._SetRho(Size)

        # set the conductances
        self.Conductances = self._CalConductance()

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "\n==> Oblea Memristor Model ..."
            print(Msg)

            # display the information
            Msg = "...%-25s: RhoMin = %.8g, RhoMax = %.8g, Rmin = %.10g, Rmax = %.10g" % \
                    (FunctionName, self.RhoMin, self.RhoMax, self.Rmin, self.Rmax)
            print(Msg)

            # display the information
            Msg = "...%-25s: Percent = %.2g, VRead = %.2g, Vth = %.4g, Rho = %s" % \
                    (FunctionName, self.Percent, self.VRead, self.Vth, str(Size))
            print(Msg)

            # display the information
            Msg = "...%-25s: Gmin = %.10g, Gmax = %.10g" % (FunctionName, \
                    self.Gmin, self.Gmax)
            print(Msg)

            # display the information
            Msg = "...%-25s: Ideal = %s, Verbose = %s, Decay Effect = %s" % \
                    (FunctionName, str(self.Ideal), str(self.Verbose), str(self.DecayEffect))
            print(Msg)

            # # display the information
            # Msg = "...%-25s: ThresholdFlag = %s" % (FunctionName, str(ThresholdFlag))
            # print(Msg)
