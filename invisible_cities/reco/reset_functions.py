import numpy  as np
import invisible_cities.reco.corrections    as corrf


def CreateVoxels(DataSiPM, sens_id, sens_q, point_dist, sipm_thr, sizeX, sizeY):
    voxX = []
    voxY = []
    voxE = []
    dist = point_dist
    selC = sens_q > sipm_thr
    for x in np.arange(DataSiPM.X[sens_id].values.min()-dist, DataSiPM.X[sens_id].values.max()+dist, sizeX):
        for y in np.arange(DataSiPM.Y[sens_id].values.min()-dist, DataSiPM.Y[sens_id].values.max()+dist, sizeY):
            voxX.append(x)
            voxY.append(y)
            voxE.append(sens_q.mean())
    return np.array([np.array(voxX), np.array(voxY), np.array(voxE)])

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

def ComputeAnodeForward(DataSiPM, vox, anode_response, sipm_dist, sipm_xy_map):
    anodeForward = []
    for sensor in anode_response:
        selVox = (np.abs(vox[0] - DataSiPM.X[sensor[0]])<=sipm_dist) & (np.abs(vox[1] - DataSiPM.Y[sensor[0]])<=sipm_dist)
        voxDX = vox[0][selVox] - DataSiPM.X[sensor[0]]
        voxDY = vox[1][selVox] - DataSiPM.Y[sensor[0]]
        anodeForward.append(np.sum(vox[2][selVox]*(1./sipm_xy_map(voxDX, voxDY).value)))
    return np.array(anodeForward)

def ComputeCathForward(vox, cath_response, pmt_xy_map):
    cathForward = []
    for sensor in range(len(cath_response)):
        cathForward.append(np.sum(vox[2]*(1./pmt_xy_map[sensor](vox[0], vox[1]).value)))
    return np.array(cathForward)

def MLEM_step(DataSiPM, oldVox, anode_response, cath_response, pmt_xy_map, sipm_xy_map, sipm_dist=20., eThres=0., fCathode = True, fAnode = True):
    newVoxE = []
    newVoxX = []
    newVoxY = []
    if fAnode:
        anodeForward = ComputeAnodeForward(DataSiPM, oldVox, anode_response, sipm_dist, sipm_xy_map)    
    if fCathode:
        cathForward = ComputeCathForward(oldVox, cath_response, pmt_xy_map)
    for j in range(len(oldVox[0])):
        efficiency = 0
        anWeight = 0
        cathWeight = 0

        if fAnode:
            voxDX = oldVox[0][j] - DataSiPM.X[anode_response[:,0]].values
            voxDY = oldVox[1][j] - DataSiPM.Y[anode_response[:,0]].values
            selV = (np.abs(voxDX) <= sipm_dist)
            selV = selV & (np.abs(voxDY) <= sipm_dist)
            anWeight += np.sum(anode_response[selV,1]*(1./sipm_xy_map(voxDX[selV], voxDY[selV]).value)/anodeForward[selV])
            efficiency += np.sum(1./sipm_xy_map(voxDX[selV], voxDY[selV]).value)
        if fCathode:
            pmtCorr = [1./pmt_xy_map[i](oldVox[0][j],oldVox[1][j]).value for i in range(len(cath_response))]
            cathWeight += np.sum(cath_response*pmtCorr/cathForward)
            efficiency += np.sum(pmtCorr)
    
        newValue = oldVox[2][j]*(anWeight+cathWeight)/efficiency

        if(newValue > eThres):
            newVoxE.append(newValue)
            newVoxX.append(oldVox[0][j])
            newVoxY.append(oldVox[1][j])
            
    return np.array([np.array(newVoxX), np.array(newVoxY), np.array(newVoxE)])
