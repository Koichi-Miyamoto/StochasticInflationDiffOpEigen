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

# customGridはDirichletなら端まで、Neumannなら端のひとつ先まで
# 戻り値は、len(customGrid)=nとして、(n-2)×(n-2)行列
def Deriv1st1dimFiniteDiff_CustomGrid(customGrid, upperBoundCond, lowerBoundCond):

    if not np.any(upperBoundCond == boundCondList):
        raise ValueError("Unknown upper boundary condition:" + upperBoundCond)
    if not np.any(lowerBoundCond == boundCondList):
        raise ValueError("Unknown lower boundary condition:" + lowerBoundCond)

    nGrid = len(customGrid) - 2
    ret = np.zeros((nGrid, nGrid))
    for iRow in range(1, nGrid - 1):
        ret[iRow, iRow - 1] = -1.0 / (customGrid[iRow + 2] - customGrid[iRow])
        ret[iRow, iRow + 1] = 1.0 / (customGrid[iRow + 2] - customGrid[iRow])

    if lowerBoundCond == "Dirichlet":
        ret[0, 1] = 1 / (customGrid[2] - customGrid[0])

    if upperBoundCond == "Dirichlet":
        ret[-1, -2] = -1 / (customGrid[-1] - customGrid[-2])
    
    return ret

def Deriv2nd1dimFiniteDiff_CustomGrid(customGrid, upperBoundCond, lowerBoundCond):

    if not np.any(upperBoundCond == boundCondList):
        raise ValueError("Unknown upper boundary condition:" + upperBoundCond)
    if not np.any(lowerBoundCond == boundCondList):
        raise ValueError("Unknown lower boundary condition:" + lowerBoundCond)
    
    nGrid = len(customGrid) - 2
    ret = np.zeros((nGrid, nGrid))
    for iRow in range(1, nGrid - 1):
        delUp = customGrid[iRow + 2] - customGrid[iRow + 1]
        delDn = customGrid[iRow + 1] - customGrid[iRow]
        ret[iRow, iRow - 1] = 2.0 / delDn / (delUp + delDn)
        ret[iRow, iRow + 1] = 2.0 / delUp / (delUp + delDn)
        ret[iRow, iRow] = -2.0 / delUp / delDn

    delUpLB = customGrid[2] - customGrid[1]
    delDnLB = customGrid[1] - customGrid[0]

    if lowerBoundCond == "Dirichlet":
        ret[0, 0] = -2.0 / delUpLB / delDnLB
        ret[0, 1] = 2.0 / delDnLB / (delUpLB + delDnLB)
    
    if lowerBoundCond == "Neumann0":
        ret[0, 0] = -2.0 / delUpLB / delDnLB
        ret[0, 1] = 2.0 / delUpLB / delDnLB

    delUpUB = customGrid[-1] - customGrid[-2]
    delDnUB = customGrid[-2] - customGrid[-3]

    if upperBoundCond == "Dirichlet":
        ret[-1, -1] = -2.0 / delUpUB / delDnUB
        ret[-1, -2] = 2.0 / delDnUB / (delUpUB + delDnUB)

    if upperBoundCond == "Neumann0":
        ret[-1, -1] = -2.0 / delUpUB / delDnUB
        ret[-1, -2] = 2.0 / delUpUB / delDnUB
    
    return ret
