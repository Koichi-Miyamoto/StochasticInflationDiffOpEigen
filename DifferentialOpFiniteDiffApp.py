import numpy as np

boundCondList = np.array([
    "Dirichlet",
    "Neumann0" # 1st derivative at boundaries = 0
    ])

def Deriv1st1dimFiniteDiff(nGrid, gridWidth, upperBoundCond, lowerBoundCond):

    if not np.any(upperBoundCond == boundCondList):
        raise ValueError("Unknown upper boundary condition:" + upperBoundCond)
    if not np.any(lowerBoundCond == boundCondList):
        raise ValueError("Unknown lower boundary condition:" + lowerBoundCond)

    ret = np.zeros((nGrid, nGrid))
    for iRow in range(1, nGrid - 1):
        ret[iRow, iRow - 1] = -0.5 / gridWidth
        ret[iRow, iRow + 1] = 0.5 / gridWidth

    if lowerBoundCond == "Dirichlet":
        ret[0, 1] = 0.5 / gridWidth

    if upperBoundCond == "Dirichlet":
        ret[-1, -2] = -0.5 / gridWidth
    
    return ret

def Deriv2nd1dimFiniteDiff(nGrid, gridWidth, upperBoundCond, lowerBoundCond):

    if not np.any(upperBoundCond == boundCondList):
        raise ValueError("Unknown upper boundary condition:" + upperBoundCond)
    if not np.any(lowerBoundCond == boundCondList):
        raise ValueError("Unknown lower boundary condition:" + lowerBoundCond)
    
    ret = np.zeros((nGrid, nGrid))
    for iRow in range(1, nGrid - 1):
        ret[iRow, iRow - 1] = ret[iRow, iRow + 1] = 1 / gridWidth / gridWidth
        ret[iRow, iRow] = -2.0 / gridWidth / gridWidth

    if lowerBoundCond == "Dirichlet":
        ret[0, 0] = -2.0 / gridWidth / gridWidth
        ret[0, 1] = 1 / gridWidth / gridWidth
    
    if lowerBoundCond == "Neumann0":
        ret[0, 0] = -2.0 / gridWidth / gridWidth
        ret[0, 1] = 2.0 / gridWidth / gridWidth

    if upperBoundCond == "Dirichlet":
        ret[-1, -1] = -2.0 / gridWidth / gridWidth
        ret[-1, -2] = 1 / gridWidth / gridWidth

    if upperBoundCond == "Neumann0":
        ret[-1, -1] = -2.0 / gridWidth / gridWidth
        ret[-1, -2] = 2.0 / gridWidth / gridWidth
    
    return ret
