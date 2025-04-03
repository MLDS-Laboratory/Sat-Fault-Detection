from Basilisk.utilities import RigidBodyKinematics as rbk
import numpy as np

class FlightModes():
    def __init__(self, sim,  startMode, switches : dict, samplingTime, taskName, **kwargs):
        self.switches = switches
        self.currentMode = startMode
        nav = kwargs.get("nav")
        if nav:
            self.navAttLog = nav.attOutMsg.recorder(samplingTime)
            self.navTransLog = nav.transOutMsg.recorder(samplingTime)
            sim.AddModelToTask(taskName, self.navAttLog)
            sim.AddModelToTask(taskName, self.navTransLog)
        spice = kwargs.get("spice")
        if spice:
            self.spiceLog = spice.planetStateOutMsgs[0].recorder(samplingTime)
            sim.AddModelToTask(taskName, self.spiceLog)
        inertial = kwargs.get("inertial")
        if inertial:
            self.inertial = inertial
    def modeManager(self, currentTime):
        if self.switches and self.switches[0]["time"] == currentTime:
            self.currentMode = self.switches[0]["mode"]
        eval("self." + self.currentMode)()
    def sunPoint(self):
        try:
            DCM = rbk.MRP2C(self.navAttLog.sigma_BN[-1])
        except:
            raise ValueError("simpleNav must be passed to run a pointing mode. ")
        sun = self.navAttLog.vehSunPntBdy[-1]
        sunInertial = DCM.T @ sun
        theta = np.atan(sunInertial[1] / sunInertial[0])
        phi = np.atan(sunInertial[2] / np.sqrt(sunInertial[0]**2 + sunInertial[1]**2)) * -1
        psi = 0
        euler = [theta, phi, psi]
        orientation = rbk.euler3212MRP(euler)
        try:
            self.inertial.sigma_R0N = orientation
        except:
            raise ValueError("inertial3D must be passed to run a pointing mode")
    def nadirPoint(self):
        try:
            scPos = self.navTransLog.r_BN_N[-1]
            DCM = rbk.MRP2C(self.navAttLog.sigma_BN[-1])
        except:
            raise ValueError("simpleNav must be passed to run a pointing mode. ")
        try:
            earthPos = self.spiceLog.PositionVector[-1]
        except:
            raise ValueError("SPICE must be passed to run a nadir pointing mode. ")
        point = (earthPos - scPos) / np.linalg.norm(earthPos - scPos)
        theta = np.atan2(point[1], point[0])
        phi = np.atan2(point[2], np.sqrt(point[0]**2 + point[1]**2)) * -1
        psi = 0
        euler = [theta, phi, psi]
        orientation = rbk.euler3212MRP(euler)
        try:
            self.inertial.sigma_R0N = orientation
        except:
            raise ValueError("inertial3D must be passed to run a pointing mode")