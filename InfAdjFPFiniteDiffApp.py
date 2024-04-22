import numpy as np
import itertools
from DifferentialOpFiniteDiffApp import Deriv1st1dimFiniteDiff, Deriv2nd1dimFiniteDiff, Deriv1st1dimFiniteDiff_CustomGrid, Deriv2nd1dimFiniteDiff_CustomGrid

def InfAdjFPFiniteDiff(
        dim,
        infPotentialFunc,
        infPotentialDerivFuncs,
        infPotentialDeriv2Funcs,
        nGridVec,
        upperBoundVec,
        lowerBoundVec,
        upperBoundCondVec,
        lowerBoundCondVec,
        logGrid=False,
        customGrid=None,
        set0OutOfInfRegion=True):

    identityMats = []

    gridsEachDim = []
    gridWidthVec = []
    upperBoundCondVecMod = ["Neumann0" if cond == "Neumann0OriFunc" else cond for cond in upperBoundCondVec]
    lowerBoundCondVecMod = ["Neumann0" if cond == "Neumann0OriFunc" else cond for cond in lowerBoundCondVec]
    for i in range(dim):
        ubCond = upperBoundCondVecMod[i]
        lbCond = lowerBoundCondVecMod[i]
        inclUb = ubCond == "Neumann0"
        inclLb = lbCond == "Neumann0"
        nGridTemp = nGridVec[i] + (not inclLb) + (not inclUb)
        if customGrid is not None:
            gridsi = customGrid[i][1:-1]
        elif logGrid:
            gridsi = np.linspace(np.log(lowerBoundVec[i]), np.log(upperBoundVec[i]), nGridTemp, endpoint=True)[int(not inclLb):(nGridTemp - int(not inclUb))]
        else:
            gridsi = np.linspace(lowerBoundVec[i], upperBoundVec[i], nGridTemp, endpoint=True)[int(not inclLb):(nGridTemp - int(not inclUb))]
        identityMats += [np.eye(len(gridsi))]
        gridsEachDim.append(gridsi)
        gridWidthVec.append(gridsi[1] - gridsi[0])
        
    grids = [np.array(g) for g in itertools.product(*gridsEachDim)]
    nGridTot = len(grids)

    # If 2nd deriv funcs are given as a 1D array, convert them to a diagonal matrix
    infPotentialDeriv2FuncMat = np.array(infPotentialDeriv2Funcs)
    if infPotentialDeriv2FuncMat.ndim == 1:
        infPotentialDeriv2FuncMat = np.full((dim, dim), lambda x: 0)
    for i in range(dim): infPotentialDeriv2FuncMat[i, i] = infPotentialDeriv2Funcs[i]

    if logGrid:
        infPotentialFuncMod = lambda g: infPotentialFunc(np.exp(g))
        infPotentialDerivFuncsMod = [lambda g: f(np.exp(g)) for f in infPotentialDerivFuncs]
        infPotentialDeriv2FuncMatMod = np.full((dim, dim), lambda x: 0)
        for i1, i2 in np.ndindex((dim, dim)):
            infPotentialDeriv2FuncMatMod[i1, i2] = lambda g: infPotentialDeriv2FuncMat[i1, i2](np.exp(g))
    else:
        infPotentialFuncMod = infPotentialFunc
        infPotentialDerivFuncsMod = infPotentialDerivFuncs
        infPotentialDeriv2FuncMatMod = infPotentialDeriv2FuncMat

    vAtGrids = np.array([infPotentialFuncMod(g) for g in grids]) / (24 * np.pi * np.pi)
    vDerAtGrids = np.array([np.array([vDerFunc(g) for g in grids]) / (24 * np.pi * np.pi) for vDerFunc in infPotentialDerivFuncsMod])
    vDer2AtGrids = np.zeros((dim, dim, nGridTot))
    for i in range(dim):
        for j in range(dim):
            vDer2AtGrids[i,j] = np.array([infPotentialDeriv2FuncMatMod[i,j](g) for g in grids]) / (24 * np.pi * np.pi)

    mat2ndDeriv = np.zeros((nGridTot, nGridTot))
    for i in range(dim):
        mat2ndDerivith = 1
        
        for j in range(dim):            
            if j == i:
                if customGrid is not None:
                    matTemp = Deriv2nd1dimFiniteDiff_CustomGrid(customGrid[i], upperBoundCondVecMod[i], lowerBoundCondVecMod[i])
                else:
                    matTemp = Deriv2nd1dimFiniteDiff(nGridVec[i], gridWidthVec[i], upperBoundCondVecMod[i], lowerBoundCondVecMod[i])
                    if logGrid: matTemp = np.diag(1 / np.exp(gridsEachDim[i]) ** 2) @ matTemp
            else:
                matTemp = identityMats[j]          

            mat2ndDerivith = np.kron(mat2ndDerivith, matTemp)
        
        mat2ndDeriv += mat2ndDerivith

        if upperBoundCondVec[i] == "Neumann0OriFunc" or lowerBoundCondVec[i] == "Neumann0OriFunc":
            mat2ndDerivithAdditional = 1

            for j in range(dim):
                if j == i:
                    matTemp = np.zeros((nGridVec[i], nGridVec[i]))
                    if upperBoundCondVec[i] == "Neumann0OriFunc": matTemp[-1,-1] = 1 / (gridsEachDim[i][-1] - gridsEachDim[i][-2])
                    if lowerBoundCondVec[i] == "Neumann0OriFunc": matTemp[0,0] = 1 / (gridsEachDim[i][1] - gridsEachDim[i][0])
                else:
                    matTemp = identityMats[j]

                mat2ndDerivithAdditional = np.kron(mat2ndDerivithAdditional, matTemp)
            
            mat2ndDerivithAdditional = -np.diag((1 + 1 / vAtGrids) * vDerAtGrids[i] / vAtGrids) @ mat2ndDerivithAdditional
            mat2ndDeriv += mat2ndDerivithAdditional

    ret = -np.diag(vAtGrids) @ mat2ndDeriv

    mat1stDeriv = np.zeros((nGridTot, nGridTot))
    for i in range(dim):
        mat1stDerivith = 1
        
        for j in range(dim):            
            if j == i:
                if customGrid is not None:
                    matTemp = Deriv1st1dimFiniteDiff_CustomGrid(customGrid[i], upperBoundCondVecMod[i], lowerBoundCondVecMod[i])
                else:
                    matTemp = Deriv1st1dimFiniteDiff(nGridVec[i], gridWidthVec[i], upperBoundCondVecMod[i], lowerBoundCondVecMod[i])
            else:
                matTemp = identityMats[j]
            mat1stDerivith = np.kron(mat1stDerivith, matTemp)

        if upperBoundCondVec[i] == "Neumann0OriFunc" or lowerBoundCondVec[i] == "Neumann0OriFunc":
            mat1stDerivithAdditional = 1

            for j in range(dim):
                if j == i:
                    matTemp = np.zeros((nGridVec[i], nGridVec[i]))
                    if upperBoundCondVec[i] == "Neumann0OriFunc": matTemp[-1,-1] = 1
                    if lowerBoundCondVec[i] == "Neumann0OriFunc": matTemp[0,0] = 1
                else:
                    matTemp = identityMats[j]

                mat1stDerivithAdditional = np.kron(mat1stDerivithAdditional, matTemp)
            
            mat1stDerivithAdditional = -np.diag(0.5 * (1 + 1 / vAtGrids) * vDerAtGrids[i] / vAtGrids) @ mat1stDerivithAdditional
            mat1stDerivith += mat1stDerivithAdditional
        
        if logGrid:
            preFancVals = -vDerAtGrids[i] / [np.exp(g[i]) for g in grids] + vAtGrids[i] / [np.exp(g[i]) ** 2 for g in grids]
        else:
            preFancVals = -vDerAtGrids[i]
        mat1stDeriv += np.diag(preFancVals) @ mat1stDerivith

    ret += mat1stDeriv

    nonDerivTerm = np.zeros(nGridTot)
    for i in range(dim):
        nonDerivTerm -= (2 * vAtGrids**2 * (1 + vAtGrids) * vDer2AtGrids[i, i] - \
                         (1 + 4 * vAtGrids + vAtGrids**2) * vDerAtGrids[i]**2) / (4 * vAtGrids**3)

    ret += np.diag(nonDerivTerm)

    # If the slowroll condition does not hold, set corresponding row and col to 0
    if set0OutOfInfRegion:
        for i1, i2 in np.ndindex((dim, dim)):
            vDer2AtGrids[i1, i2, :] = [infPotentialDeriv2FuncMatMod[i1, i2](g) / (24 * np.pi * np.pi) for g in grids]
        for ig in range(nGridTot):
            vDers = vDerAtGrids[:,ig]
            slowrollEps = 0.5 * np.sum((vDers / vAtGrids[ig]) ** 2)
            slowrollEta = np.abs(vDers.T @ vDer2AtGrids[:,:,ig] @ vDers / vAtGrids[ig] / np.sum(vDers ** 2))
            if slowrollEps > 1 or slowrollEta > 1:
                ret[ig,:] = 0
                ret[:,ig] = 0
    
    return ret

def InfAdjFPFiniteDiff_Larsson(
        dim,
        infPotentialFunc,
        infPotentialDerivFuncs,
        infPotentialDeriv2Funcs,
        nGrid,
        upperBound,
        lowerBound,
        upperBoundCond="Dirichlet",
        lowerBoundCond="Dirichlet"):
    
    # assume that all upper bounds are equal and all lower bounds are equal

    nGridTot = nGrid ** dim
    gridWidth = (upperBound - lowerBound) / (nGrid + 1)

    gridsEachDim = []
    idVecLikeTemp = []
    unitVecs = []
    inclUb = upperBoundCond == "Neumann0"
    inclLb = lowerBoundCond == "Neumann0"
    nGridTemp = nGrid + (not inclLb) + (not inclUb)
    gridsOneDim = np.linspace(lowerBound, upperBound, nGridTemp, endpoint=True)[int(not inclLb):(nGridTemp - int(not inclUb))]
    for i in range(dim):
        gridsEachDim.append(gridsOneDim)
        idVecLikeTemp.append(np.arange(nGrid))
        unitVecs.append(np.eye(dim)[i])
        
    grids = [np.array(g) for g in itertools.product(*gridsEachDim)]
    idVecLikes = [np.array(i) for i in itertools.product(*idVecLikeTemp)]

    # If 2nd deriv funcs are given as a 1D array, convert them to a diagonal matrix
    infPotentialDeriv2FuncMat = np.array(infPotentialDeriv2Funcs)
    if infPotentialDeriv2FuncMat.ndim == 1:
        infPotentialDeriv2FuncMat = np.full((dim, dim), lambda x: 0)
        for i in range(dim): infPotentialDeriv2FuncMat[i, i] = infPotentialDeriv2Funcs[i]

    ret = np.zeros((nGridTot, nGridTot))
    for idVecLike in idVecLikes:
        i = ConvIdVecLikeToId(idVecLike, dim, nGrid)
        grid = grids[i]
        v = infPotentialFunc(grid) / (24 * np.pi * np.pi)
        vDers = np.array([vDerFunc(grid) / (24 * np.pi * np.pi) for vDerFunc in infPotentialDerivFuncs])
        vDer2s = np.zeros((dim, dim))
        for i1, i2 in np.ndindex((dim, dim)):
            vDer2s[i1, i2] = infPotentialDeriv2FuncMat[i1, i2](grid) / (24 * np.pi * np.pi)

        # slow-roll parameters
        slowrollEps = 0.5 * np.sum((vDers / v) ** 2)
        slowrollEta = np.abs(vDers.T @ vDer2s @ vDers / v / np.sum(vDers ** 2))

        # Fill the discretizing matrix for the differential operator if and only if the slowroll conditions hold.
        # If not, the corresponding entry is left zero.
        if slowrollEps <= 1 and slowrollEta <= 1:
            nonDerivTerm = 0
            coefDiag = 0
            for iDim in range(dim):
                ei = unitVecs[iDim]
                nonDerivTerm -= (2 * v * v * (1 + v) * vDer2s[iDim, iDim] - (1 + 4 * v + v * v) * vDers[iDim] * vDers[iDim]) / (4 * v * v * v)
                if idVecLike[iDim] == 0 and lowerBoundCond == "Neumann0":
                    coefDiag += 2 * v
                    ret[i, i + nGrid ** (dim - 1 - iDim)] = -2 * v / gridWidth ** 2
                elif idVecLike[iDim] == nGrid - 1 and upperBoundCond == "Neumann0":
                    coefDiag += 2 * v
                    ret[i, i - nGrid ** (dim - 1 - iDim)] = -2 * v / gridWidth ** 2
                else:
                    vPlus = infPotentialFunc(grid + 0.5 * gridWidth * ei) / (24 * np.pi * np.pi)
                    vMinus = infPotentialFunc(grid - 0.5 * gridWidth * ei) / (24 * np.pi * np.pi)
                    if idVecLike[iDim] < nGrid - 1:
                        ret[i, i + nGrid ** (dim - 1 - iDim)] = -vPlus / gridWidth ** 2
                    if idVecLike[iDim] == 0 and lowerBoundCond == "Neumann0":
                        ret[i, i + nGrid ** (dim - 1 - iDim)] += -vPlus / gridWidth ** 2
                    if idVecLike[iDim] > 0: ret[i, i - nGrid ** (dim - 1 - iDim)] = ret[i - nGrid ** (dim - 1 - iDim), i]
                    coefDiag += vPlus + vMinus
                
            ret[i, i] = coefDiag / gridWidth ** 2 + nonDerivTerm
        else:
            ret[:i, i] = 0
    
    return ret

def InfAdjFPFiniteDiff_OriginalOperator(
        dim,
        infPotentialFunc,
        infPotentialDerivFuncs,
        nGridVec,
        upperBoundVec,
        lowerBoundVec,
        upperBoundCondVec,
        lowerBoundCondVec,
        customGrid=None):

    identityMats = []

    gridsEachDim = []
    gridWidthVec = []
    for i in range(dim):
        ubCond = upperBoundCondVec[i]
        lbCond = lowerBoundCondVec[i]
        inclUb = ubCond == "Neumann0"
        inclLb = lbCond == "Neumann0"
        nGridTemp = nGridVec[i] + (not inclLb) + (not inclUb)
        if customGrid is not None:
            gridsi = customGrid[i][1:-1]
        else:
            gridsi = np.linspace(lowerBoundVec[i], upperBoundVec[i], nGridTemp, endpoint=True)[int(not inclLb):(nGridTemp - int(not inclUb))]
        identityMats += [np.eye(len(gridsi))]
        gridsEachDim.append(gridsi)
        gridWidthVec.append(gridsi[1] - gridsi[0])
        
    grids = [np.array(g) for g in itertools.product(*gridsEachDim)]
    nGridTot = len(grids)

    vAtGrids = np.array([infPotentialFunc(g) for g in grids]) / (24 * np.pi * np.pi)
    vDerAtGrids = np.array([np.array([vDerFunc(g) for g in grids]) / (24 * np.pi * np.pi) for vDerFunc in infPotentialDerivFuncs])

    mat2ndDeriv = np.zeros((nGridTot, nGridTot))
    for i in range(dim):
        mat2ndDerivith = 1
        
        for j in range(dim):            
            if j == i:
                if customGrid is not None:
                    matTemp = Deriv2nd1dimFiniteDiff_CustomGrid(customGrid[i], upperBoundCondVec[i], lowerBoundCondVec[i])
                else:
                    matTemp = Deriv2nd1dimFiniteDiff(nGridVec[i], gridWidthVec[i], upperBoundCondVec[i], lowerBoundCondVec[i])
            else:
                matTemp = identityMats[j]
            mat2ndDerivith = np.kron(mat2ndDerivith, matTemp)
        
        mat2ndDeriv += mat2ndDerivith

    ret = -np.diag(vAtGrids) @ mat2ndDeriv

    mat1stDeriv = np.zeros((nGridTot, nGridTot))
    for i in range(dim):
        mat1stDerivith = 1
        
        for j in range(dim):            
            if j == i:
                if customGrid is not None:
                    matTemp = Deriv1st1dimFiniteDiff_CustomGrid(customGrid[i], upperBoundCondVec[i], lowerBoundCondVec[i])
                else:
                    matTemp = Deriv1st1dimFiniteDiff(nGridVec[i], gridWidthVec[i], upperBoundCondVec[i], lowerBoundCondVec[i])
            else:
                matTemp = identityMats[j]
            mat1stDerivith = np.kron(mat1stDerivith, matTemp)
        
        preFancVals = vDerAtGrids[i] / vAtGrids
        mat1stDeriv += np.diag(preFancVals) @ mat1stDerivith

    ret += mat1stDeriv
    
    return ret

def ConvIdVecLikeToId(idVecLike, dim, nGrid):
    return np.dot(idVecLike, nGrid ** np.arange(dim - 1, -1, -1))