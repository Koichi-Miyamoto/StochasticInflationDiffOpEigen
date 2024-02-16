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

    ret = np.zeros((nGridTot, nGridTot))
    for idVecLike in idVecLikes:
        i = ConvIdVecLikeToId(idVecLike, dim, nGrid)
        grid = grids[i]
        v = infPotentialFunc(grid) / (24 * np.pi * np.pi)
        nonDerivTerm = 0
        coefDiag = 0
        for iDim in range(dim):
            ei = unitVecs[iDim]
            vPlus = infPotentialFunc(grid + 0.5 * gridWidth * ei) / (24 * np.pi * np.pi)
            vMinus = infPotentialFunc(grid - 0.5 * gridWidth * ei) / (24 * np.pi * np.pi)
            if idVecLike[iDim] < nGrid - 1: ret[i, i + nGrid ** (dim - 1 - iDim)] = -vPlus / gridWidth ** 2
            if idVecLike[iDim] > 0: ret[i, i - nGrid ** (dim - 1 - iDim)] = -vMinus / gridWidth ** 2
            coefDiag += vPlus + vMinus
            vDer = infPotentialDerivFuncs[iDim](grid) / (24 * np.pi * np.pi)
            vDer2 = infPotentialDeriv2Funcs[iDim](grid) / (24 * np.pi * np.pi)
            nonDerivTerm -= (2 * v * v * (1 + v) * vDer2 - (1 + 4 * v + v * v) * vDer * vDer) / (4 * v * v * v)
        ret[i, i] = coefDiag / gridWidth ** 2 + nonDerivTerm
    
    return ret

def ConvIdVecLikeToId(idVecLike, dim, nGrid):
    return np.dot(idVecLike, nGrid ** np.arange(dim - 1, -1, -1))