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
        gridWidthVec,
        upperBoundCondVec,
        lowerBoundCondVec,
        hermitianize=False):

    identityMats = [np.eye(n) for n in nGridVec]
    nGridTot = np.prod(nGridVec)

    gridsEachDim = []
    for i in range(dim):
        ubCond = upperBoundCondVec[i]
        lbCond = lowerBoundCondVec[i]
        inclUb = ubCond == "Neumann0"
        inclLb = lbCond == "Neumann0"
        nGridTemp = nGridVec[i] + (not inclLb)
        gridsEachDim.append(np.linspace(lowerBoundVec[i], upperBoundVec[i], nGridTemp, endpoint=inclUb)[int(not inclLb):])
        
    grids = [np.array(g) for g in itertools.product(*gridsEachDim)]
    vAtGrids = np.array([infPotentialFunc(g) for g in grids]) / (24 * np.pi * np.pi)

    ret = np.zeros(nGridTot)
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

    mat1stDeriv = np.zeros(nGridTot)
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