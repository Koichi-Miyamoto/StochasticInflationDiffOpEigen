import numpy as np
import itertools
from DifferentialOpFiniteDiffApp import Deriv1st1dimFiniteDiff, Deriv2nd1dimFiniteDiff

def InfAdjFPFiniteDiff(
        dim,
        infPotentialFunc,
        infPotentialDerivFuncs,
        nGridVec,
        upperBoundVec,
        lowerBoundVec,
        upperBoundCondVec,
        lowerBoundCondVec,
        hermitianize=False):

    identityMats = [np.eye(n) for n in nGridVec]
    nGridTot = np.prod(nGridVec)

    gridsEachDim = []
    gridWidthVec = []
    for i in range(dim):
        ubCond = upperBoundCondVec[i]
        lbCond = lowerBoundCondVec[i]
        inclUb = ubCond == "Neumann0"
        inclLb = lbCond == "Neumann0"
        nGridTemp = nGridVec[i] + (not inclLb) + (not inclUb)
        gridsEachDim.append(np.linspace(lowerBoundVec[i], upperBoundVec[i], nGridTemp, endpoint=True)[int(not inclLb):(nGridTemp - int(not inclUb))])
        gridWidthVec.append((upperBoundVec[i] - lowerBoundVec[i]) / (nGridTemp - 1))
        
    grids = [np.array(g) for g in itertools.product(*gridsEachDim)]
    vAtGrids = np.array([infPotentialFunc(g) for g in grids]) / (24 * np.pi * np.pi)

    mat2ndDeriv = np.zeros((nGridTot, nGridTot))
    for i in range(dim):
        mat2ndDerivith = 1
        
        for j in range(dim):            
            if j == i:
                matTemp = Deriv2nd1dimFiniteDiff(nGridVec[i], gridWidthVec[i], upperBoundCondVec[i], lowerBoundCondVec[i])
            else:
                matTemp = identityMats[j]
            mat2ndDerivith = np.kron(mat2ndDerivith, matTemp)
        
        mat2ndDeriv += mat2ndDerivith

    ret = np.diag(vAtGrids) @ mat2ndDeriv

    mat1stDeriv = np.zeros((nGridTot, nGridTot))
    for i in range(dim):
        mat1stDerivith = 1
        
        for j in range(dim):            
            if j == i:
                matTemp = Deriv1st1dimFiniteDiff(nGridVec[i], gridWidthVec[i], upperBoundCondVec[i], lowerBoundCondVec[i])
            else:
                matTemp = identityMats[j]
            mat1stDerivith = np.kron(mat1stDerivith, matTemp)
        
        viAtGrids = [infPotentialDerivFuncs[i](g) for g in grids]
        mat1stDeriv += np.diag(viAtGrids) @ mat1stDerivith

    mat1stDeriv /= -vAtGrids
    ret += mat1stDeriv

    if hermitianize:
        wSqrt = np.diag(np.exp(0.5 / viAtGrids) / np.sqrt(viAtGrids))
        ret = wSqrt @ ret @ (1 / wSqrt)
    
    return ret

def InfAdjFPFiniteDiff_Larsson(
        dim,
        infPotentialFunc,
        infPotentialDerivFuncs,
        infPotentialDeriv2Funcs,
        nGrid,
        upperBound,
        lowerBound):
    
    # support Dirichlet only
    # assume that all upper bounds are equal and all lower bounds are equal

    nGridTot = nGrid ** dim
    gridWidth = (upperBound - lowerBound) / (nGrid + 1)

    gridsEachDim = []
    idVecLikeTemp = []
    unitVecs = []
    for i in range(dim):
        gridsEachDim.append(np.linspace(lowerBound, upperBound, nGrid + 2, endpoint=True)[1:-1])
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
                vPlus = infPotentialFunc(grid + 0.5 * gridWidth * ei) / (24 * np.pi * np.pi)
                vMinus = infPotentialFunc(grid - 0.5 * gridWidth * ei) / (24 * np.pi * np.pi)
                if idVecLike[iDim] < nGrid - 1: ret[i, i + nGrid ** (dim - 1 - iDim)] = -vPlus / gridWidth ** 2
                if idVecLike[iDim] > 0: ret[i, i - nGrid ** (dim - 1 - iDim)] = ret[i - nGrid ** (dim - 1 - iDim), i]
                coefDiag += vPlus + vMinus
                nonDerivTerm -= (2 * v * v * (1 + v) * vDer2s[iDim, iDim] - (1 + 4 * v + v * v) * vDers[iDim] * vDers[iDim]) / (4 * v * v * v)
            ret[i, i] = coefDiag / gridWidth ** 2 + nonDerivTerm
        else:
            ret[:i, i] = 0
    
    return ret

def ConvIdVecLikeToId(idVecLike, dim, nGrid):
    return np.dot(idVecLike, nGrid ** np.arange(dim - 1, -1, -1))