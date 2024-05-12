import numpy as np
import torch, os
import torch.nn as nn

class BatasMemristorTorch(nn.Module):
    def __init__(self, Ron, Roff, D, uv, InitVals):

        # set the function name
        FunctionName = "Batas::__init__()"

        # memristor model name
        self.ModelName = "Batas"

        self.Ron = nn.Parameter(torch.tensor(Ron))
        self.Roff = nn.Parameter(torch.tensor(Roff))
        self.D = nn.Parameter(torch.tensor(D))
        self.uv = nn.Parameter(torch.tensor(uv))
        self.w = nn.Parameter(torch.tensor(InitVals))

    def ResetInitVals(self, InitVals):
        self.w.data = torch.tensor(InitVals)


    def UpdateVals(self, Vin):
        #Vin is externally applied voltage that is used to update the internal state 
        # variable w of the memristor mode
        self.w = self.w + Vin

    def GetInitVals(self, InitStates):
        # Assuming InitStates contains relevant initial parameters
        Ron = InitStates['Ron']
        Roff = InitStates['Roff']
        D = InitStates['D']
        r = self.Ron * self.w / self.D + self.Roff * (1 - self.w / self.D)
        return r

    def GetVals(self, VinVals):
        self.UpdateVals(VinVals)
        r = self.Ron * self.w / self.D + self.Roff * (1 - self.w / self.D)
        return r

def GetWinHomePath():
    # get the username from the environmental variables
    UserName    = os.getenv("USERNAME")
    FolderPath  = join("C:\\Users", UserName, "Documents")
    return FolderPath

if __name__ == "__main__":
    # import module
    from os.path import join
    import sys
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"figure.autolayout": True})

    # **************************************************************************
    # set the function name
    # **************************************************************************
    FunctionName = "Batas::main()"

    # **************************************************************************
    # set the model name
    # **************************************************************************
    ModelName   = "Batas"

    # **************************************************************************
    # check the platform to import mem-models
    # **************************************************************************
    Platform = sys.platform
    if (Platform == "darwin") or (Platform == "linux"):
        # set common folders
        CommonPath  = "/stash/tlab/dattransj/MemCommonFuncs"

    elif Platform == "win32":
        CommonPath  = join(GetWinHomePath(), "MemCommonFuncs")
        
    else:
        # format error message
        Msg = "unknown platform => <%s>" % (Platform)
        raise ValueError(Msg)

    # append to the system path
    sys.path.append(CommonPath)
   # import WaveGenerator

    # **************************************************************************
    # a temporary fix for OpenMP
    # **************************************************************************
    os.environ["KMP_DUPLICATE_LIB_OK"]="True"

    # Initialize the memristor with initial values
    Ron = [0.5]
    InitVals = [0.1]  # Initial value for w
    Roff = [0.2]
    D = [0.3]
    uv =[[0.5]]

    memristor = BatasMemristorTorch(Ron, Roff, D, uv, InitVals)