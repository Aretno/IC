import numpy  as np
import invisible_cities.reco.corrections    as corrf
import numba
#from invisible_cities.reco.corrections.corrections import Correction

def CreateVoxels(DataSiPM, sens_id, sens_q, point_dist, sipm_thr, sizeX, sizeY, rMax):
    voxX = []
    voxY = []
    voxE = []
    dist = point_dist
    selC = sens_q > sipm_thr
    rangex  = np.arange(DataSiPM.X[sens_id].values.min()-dist, DataSiPM.X[sens_id].values.max()+dist, sizeX)
    rangey = np.arange(DataSiPM.Y[sens_id].values.min()-dist, DataSiPM.Y[sens_id].values.max()+dist, sizeY)
    for x in rangex:
        for y in rangey:
            if(np.sqrt(x*x+y*y) > rMax):
                continue
            voxX.append(x)
            voxY.append(y)
            voxE.append(sens_q.mean())
    return np.array([voxX, voxY, voxE])

def CreateSiPMresponse(DataSiPM, sens_id, sens_q, sipm_dist, sipm_thr, vox):
    sens_response = []
    selDB = (DataSiPM.X >= vox[0].min()-sipm_dist) & (DataSiPM.X <= vox[0].max()+sipm_dist) 
    selDB = selDB & (DataSiPM.Y >= vox[1].min()-sipm_dist) & (DataSiPM.Y <= vox[1].max()+sipm_dist) 
    for ID in DataSiPM[selDB].index.values:
        sel = (sens_id==ID)
        if sens_q[sel] > sipm_thr:
            sens_response.append([ID, sens_q[sel]])
        else:
            sens_response.append([ID,0.])
    return np.array(sens_response)

def ComputeCathForward(vox, cath_response, pmt_xy_map):
    cathForward = []
    for sensor in range(len(cath_response)):
        cathForward.append(sum(vox[2]*(1./pmt_xy_map[sensor](vox[0], vox[1]).value)))
    return np.array(cathForward)

@numba.jit
def pmtCorrections(j, oldVox, pmt_xy_map, n_pmts):
    pmtcorrection = np.zeros(n_pmts)
    for i in range(n_pmts):
        pmtcorrection[i] = 1./pmt_xy_map[i](oldVox[0][j],oldVox[1][j]).value
    return pmtcorrection
        
@numba.jit
def MLEM_step2(voxDX, voxDY, oldVox, selVox, selSens, anode_response, cath_response, pmt_xy_map, xy_prob, sipm_dist=20., eThres=0., fCathode = True, fAnode = True):
    newVoxE = []
    newVoxX = []
    newVoxY = []

    anodeForward = 0
    cathForward = 0
    
    if fAnode:
        anodeForward = ComputeAnodeForward2(voxDX, voxDY, oldVox, anode_response, sipm_dist, xy_prob, selVox)    
    if fCathode:
        cathForward = ComputeCathForward(oldVox, cath_response, pmt_xy_map)
        
    for j in range(len(oldVox[0])):
        efficiency = 0
        anWeight = 0
        cathWeight = 0

        if fAnode:
            selS = selSens[j] #(np.abs(voxDX[j]) <= sipm_dist) & (np.abs(voxDY[j]) <= sipm_dist)
            if sum(anode_response[selS,1]) == 0.:
                newVoxE.append(0.)
                newVoxX.append(oldVox[0][j])
                newVoxY.append(oldVox[1][j])
                continue
            sipmCorr = xy_prob[j][selS]#1./sipm_xy_map(voxDX[j][selV], voxDY[j][selV]).value
            anWeight += sum(anode_response[selS,1]*(sipmCorr)/anodeForward[selS])
            efficiency += sum(sipmCorr)
        if fCathode:
            pmtCorr = pmtCorrections(j, oldVox, pmt_xy_map, len(cath_response))
            cathWeight += sum(cath_response*pmtCorr/cathForward)
            efficiency += sum(pmtCorr)
            
        newValue = oldVox[2][j]*(anWeight+cathWeight)/efficiency

        if(newValue >= eThres):
            newVoxE.append(newValue)
            newVoxX.append(oldVox[0][j])
            newVoxY.append(oldVox[1][j])

    return np.array([newVoxX, newVoxY, newVoxE])

def computeDiff(DataSiPM, oldVox, anode_response):
    voxD = np.array([[oldVox[0][j] - DataSiPM.X[anode_response[:,0]].values, oldVox[1][j] - DataSiPM.Y[anode_response[:,0]].values] for j in range(len(oldVox[0]))])
    return voxD[:,0], voxD[:,1]

@numba.autojit
def ComputeAnodeForward2(voxDX, voxDY, vox, anode_response, sipm_dist, xy_prob, selVox):
    dim = len(anode_response)
    anodeForward = np.zeros(dim)
    for sensor in range(dim):
        selV = selVox[sensor]
        #        selVox = (np.abs(voxDX[:,sensor]) <= sipm_dist) & (np.abs(voxDY[:,sensor]) <= sipm_dist)
        anodeForward[sensor] = (sum(vox[2][selV]*xy_prob[selV,sensor]))
    return anodeForward

def computeProb(sipm_xy_map, voxDX, voxDY):
    xyprob = [1./sipm_xy_map(voxDX[j], voxDY[j]).value for j in range(len(voxDX))]
    return np.array(xyprob)

def createSel(voxDX, voxDY, anode_response, sipm_dist):
    selVox = []
    selSens = []
    for sensor in range(len(anode_response)):
        selVox.append( (np.abs(voxDX[:,sensor]) <= sipm_dist) & (np.abs(voxDY[:,sensor]) <= sipm_dist) )
    for voxel in range(len(voxDX)):
        selSens.append( (np.abs(voxDX[voxel]) <= sipm_dist) & (np.abs(voxDY[voxel]) <= sipm_dist) )
    return selVox, selSens

def MLEM_step(voxDX, voxDY, oldVox, anode_response, cath_response, pmt_xy_map, sipm_xy_map, sipm_dist=20., eThres=0., fCathode = True, fAnode = True):
    newVoxE = []
    newVoxX = []
    newVoxY = []
    
    if fAnode:
        anodeForward = ComputeAnodeForward(voxDX, voxDY, oldVox, anode_response, sipm_dist, sipm_xy_map)    
    if fCathode:
        cathForward = ComputeCathForward(oldVox, cath_response, pmt_xy_map)
        
    for j in range(len(oldVox[0])):
        efficiency = 0
        anWeight = 0
        cathWeight = 0

        if fAnode:
            selV = (np.abs(voxDX[j]) <= sipm_dist) & (np.abs(voxDY[j]) <= sipm_dist)
            if sum(anode_response[selV,1]) == 0.:
                newVoxE.append(0.)
                newVoxX.append(oldVox[0][j])
                newVoxY.append(oldVox[1][j])
                continue
            sipmCorr = 1./sipm_xy_map(voxDX[j][selV], voxDY[j][selV]).value
            anWeight += sum(anode_response[selV,1]*(sipmCorr)/anodeForward[selV])
            efficiency += sum(sipmCorr)
        if fCathode:
            pmtCorr = pmtCorrections(j, oldVox, pmt_xy_map, len(cath_response))
            cathWeight += sum(cath_response*pmtCorr/cathForward)
            efficiency += sum(pmtCorr)
            
        newValue = oldVox[2][j]*(anWeight+cathWeight)/efficiency

        if(newValue >= eThres):
            newVoxE.append(newValue)
            newVoxX.append(oldVox[0][j])
            newVoxY.append(oldVox[1][j])

    return np.array([newVoxX, newVoxY, newVoxE])

def ComputeAnodeForward(voxDX, voxDY, vox, anode_response, sipm_dist, sipm_xy_map):
    anodeForward = []
    for sensor in range(len(anode_response)):
        selVox = (np.abs(voxDX[:,sensor]) <= sipm_dist) & (np.abs(voxDY[:,sensor]) <= sipm_dist)
        anodeForward.append(sum(vox[2][selVox]*(1./sipm_xy_map(voxDX[selVox,sensor], voxDY[selVox,sensor]).value)))
    return np.array(anodeForward)

