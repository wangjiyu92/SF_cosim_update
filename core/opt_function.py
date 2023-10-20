# ~~~~~~~~~~~~ Author: Fei Ding @ NREL ~~~~~~~~~~~~~~~
import numpy as np
# import os
# import json
# from scipy.sparse import lil_matrix
# import scipy.sparse.linalg as sp
# import scipy.sparse as sparse
import math
import csv
from . import dss_function
import matplotlib.pyplot as plt


def linear_powerflow_model(Y00, Y01, Y10, Y11_inv, I_coeff, V1, slack_no):
    # voltage linearlization
    V1_conj = np.conj(V1[slack_no:])
    V1_conj_inv = 1 / V1_conj
    coeff_V = Y11_inv * V1_conj_inv
    coeff_V_P = coeff_V
    coeff_V_Q = -1j * coeff_V
    coeff_Vm = -np.dot(Y11_inv, np.dot(Y10, V1[:slack_no]))

    # voltage magnitude linearization
    m = coeff_Vm
    m_inv = 1 / coeff_Vm
    coeff_Vmag_k = abs(m)
    A = (np.multiply(coeff_V.transpose(), m_inv)).transpose()
    coeff_Vmag_P = (np.multiply(A.real.transpose(), coeff_Vmag_k)).transpose()
    coeff_Vmag_Q = (np.multiply((-1j * A).real.transpose(), coeff_Vmag_k)).transpose()

    # current linearization
    if len(I_coeff):
        coeff_I_P = np.dot(I_coeff[:, slack_no:], coeff_V_P)
        coeff_I_Q = np.dot(I_coeff[:, slack_no:], coeff_V_Q)
        coeff_I_const = np.dot(I_coeff[:, slack_no:], coeff_Vm) + np.dot(I_coeff[:, 0:slack_no], V1[:slack_no])
        # TODO
        coeff_Psub_P, coeff_Psub_Q, coeff_Psub_const = 0, 0, 0
    else:
        coeff_I_P = []
        coeff_I_Q = []
        coeff_I_const = []

        # feeder head power linearization
        tempP = np.dot(np.conj(Y01), np.conj(coeff_V_P))
        tempQ = np.dot(np.conj(Y01), np.conj(coeff_V_Q))
        coeff_Ssub_P = (tempP.transpose() * V1[:slack_no]).transpose()
        coeff_Ssub_Q = (tempQ.transpose() * V1[:slack_no]).transpose()
        temp_const = np.dot(np.conj(Y00), np.conj(V1[:slack_no])) + np.dot(np.conj(Y01), np.conj(coeff_Vm))
        coeff_Ssub_const = (temp_const.transpose() * V1[:slack_no]).transpose()
        coeff_Psub_P = coeff_Ssub_P.real
        coeff_Psub_Q = coeff_Ssub_Q.real
        coeff_Psub_const = coeff_Ssub_const.real

    # with open(os.path.join(os.path.dirname(__file__), 'linearPF_coeff.json'), 'w') as f:
    #     json.dump([coeff_V_P, coeff_V_Q, coeff_Vm, coeff_Vmag_P, coeff_Vmag_Q, coeff_Vmag_k, coeff_I_P, coeff_I_Q,
    #         coeff_I_const, coeff_Psub_P, coeff_Psub_Q, coeff_Psub_const], f)

    return [coeff_V_P, coeff_V_Q, coeff_Vm, coeff_Vmag_P, coeff_Vmag_Q, coeff_Vmag_k, coeff_I_P, coeff_I_Q,
            coeff_I_const, coeff_Psub_P, coeff_Psub_Q, coeff_Psub_const]


def validate_linear_model(linear_PF_coeff, PQ_node, slack_number, Vmes, Imes, Pmes, Qmes, Vbase_allnode):
    coeff_V_P = linear_PF_coeff[0]
    coeff_V_Q = linear_PF_coeff[1]
    coeff_Vm = linear_PF_coeff[2]
    coeff_Vmag_P = linear_PF_coeff[3]
    coeff_Vmag_Q = linear_PF_coeff[4]
    coeff_Vmag_k = linear_PF_coeff[5]
    coeff_I_P = linear_PF_coeff[6]
    coeff_I_Q = linear_PF_coeff[7]
    coeff_I_const = linear_PF_coeff[8]
    coeff_Ssub_P = linear_PF_coeff[9]
    coeff_Ssub_Q = linear_PF_coeff[10]
    coeff_Ssub_const = linear_PF_coeff[11]
    Pbus = [np.real(ii) * 1000 for ii in PQ_node[slack_number:]]
    Qbus = [np.imag(ii) * 1000 for ii in PQ_node[slack_number:]]

    # compute linear power flow model results
    V_cal = coeff_Vm + np.dot(coeff_V_P, np.conjugate(PQ_node[slack_number:]) * 1000)
    Vmag_cal = coeff_Vmag_k + np.dot(coeff_Vmag_P, np.array(Pbus)) + np.dot(coeff_Vmag_Q, np.array(Qbus))
    Vmag_cal = list(map(lambda x: x[0] / x[1], zip(Vmag_cal, Vbase_allnode[slack_number:])))
    Ibranch_cal = coeff_I_const + np.dot(coeff_I_P, np.array(Pbus)) + np.dot(coeff_I_Q, np.array(Qbus))
    Ssub_cal = coeff_Ssub_const + np.dot(coeff_Ssub_P, np.array(Pbus)) + np.dot(coeff_Ssub_Q, np.array(Qbus))

    # compute errors between computed values and measured values
    V_diff = list(map(lambda x: abs(x[0] - x[1]) / abs(x[0]) * 100, zip(Vmes[slack_number:], list(Vmag_cal))))
    print('Voltage Magnitude Difference (%) = ' + str(sum(V_diff) / len(V_diff)))
    with open('voltage_diff.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(V_diff)
    f.close()
    plt.plot(Vmes[slack_number:])
    plt.plot(Vmag_cal)
    plt.legend(['exact', 'approximate'])
    plt.show()

    Ibranch_diff = list(map(lambda x: abs(x[0] - x[1]), zip(Imes, [abs(ii) for ii in Ibranch_cal])))
    print('Branch Current Difference (Amp) = ' + str(sum(Ibranch_diff) / len(Ibranch_diff)))
    with open('current_diff.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(Ibranch_diff)
    f.close()

    Psub_diff = abs(Pmes - sum(np.real(Ssub_cal)) / 1000)
    Qsub_diff = abs(Qmes - sum(np.imag(Ssub_cal)) / 1000)
    print('Difference in Active Power at Substation = ' + str(Psub_diff))
    print('Difference in Reactive Power at Substation = ' + str(Qsub_diff))
    return [V_cal, Vmag_cal, Ibranch_cal, Ssub_cal]


def costFun(x, dual_upper, dual_lower, v1_pu, Ppv_max, coeff_p, coeff_q, NPV, control_bus_index, Vupper, Vlower,
            dual_current, ThermalLimit, I1_mag):
    # cost_function = coeff_p*(Pmax-P)^2+coeff_q*Q^2+dual_upper*(v1-1.05)+dual_lower*(0.95-v1)
    f1 = 0
    for ii in range(NPV):
        f1 = f1 + coeff_p * (Ppv_max[ii] - x[ii]) * (Ppv_max[ii] - x[ii]) + coeff_q * x[ii + NPV] * x[ii + NPV]
    # f = f1 + np.dot(dual_upper,(np.array(v1_pu)[control_bus_index]-Vupper)) + np.dot(dual_lower,(Vlower-np.array(v1_pu)[control_bus_index]))
    v_evaluate = [v1_pu[ii] for ii in control_bus_index]
    f2 = f1 + np.dot(dual_upper, np.array([max(ii - Vupper, 0) for ii in v_evaluate])) + np.dot(dual_lower, np.array(
        [max(Vlower - ii, 0) for ii in v_evaluate]))
    f3 = np.dot(dual_current, np.array(
        [max(ii, 0) for ii in list(map(lambda x: x[0] * x[0] - x[1] * x[1], zip(I1_mag, ThermalLimit)))]))
    f = f2 + f3
    return [f1, f]


def PV_costFun_gradient(x, coeff_p, coeff_q, Pmax):
    grad = np.zeros(len(x))
    for ii in range(int(len(x) / 2)):
        grad[ii] = -2 * coeff_p * (Pmax[ii] * 1000 - x[ii] * 1000)
        grad[ii + int(len(x) / 2)] = 2 * coeff_q * x[ii + int(len(x) / 2)] * 1000
        # grad[ii + int(len(x) / 2)] = 0
    return grad


def voltage_constraint_gradient(AllNodeNames, node_withPV, dual_upper, dual_lower, coeff_Vmag_p, coeff_Vmag_q):
    node_noslackbus = AllNodeNames
    node_noslackbus[0:3] = []
    grad_upper = np.matrix([0] * len(node_noslackbus) * 2).transpose()
    grad_lower = np.matrix([0] * len(node_noslackbus) * 2).transpose()
    count = 0
    for node in node_noslackbus:
        if node in node_withPV:
            grad_upper[count] = dual_upper.transpose() * coeff_Vmag_p[:, count]
            grad_upper[count + len(node_noslackbus)] = dual_upper.transpose() * coeff_Vmag_q[:, count]
            grad_lower[count] = -dual_lower.transpose() * coeff_Vmag_p[:, count]
            grad_lower[count + len(node_noslackbus)] = -dual_lower.transpose() * coeff_Vmag_q[:, count]
        count = count + 1
    return [grad_upper, grad_lower]


def current_constraint_gradient(AllNodeNames, node_withPV, dual_upper, coeff_Imag_p, coeff_Imag_q):
    node_noslackbus = AllNodeNames
    node_noslackbus[0:3] = []
    grad_upper = np.matrix([0] * len(node_noslackbus) * 2).transpose()
    count = 0
    for node in node_noslackbus:
        if node in node_withPV:
            grad_upper[count] = dual_upper.transpose() * coeff_Imag_p[:, count]
            grad_upper[count + len(node_noslackbus)] = dual_upper.transpose() * coeff_Imag_q[:, count]
        count = count + 1
    return grad_upper


def voltage_constraint(V1_mag):
    g = V1_mag - 1.05
    g.append(0.95 - V1_mag)
    return g


def current_constraint(I1_mag, Imax):
    g = []
    g.append(I1_mag - Imax)
    return g


def project_dualvariable(mu):
    for ii in range(len(mu)):
        mu[ii] = max(mu[ii], 0)
    return mu


def project_PV(x, Pmax, Sinv):
    Qavailable = 0
    Pavailable = 0
    if x[0] > Pmax:
        x[0] = Pmax
    elif x[0] < 0:
        x[0] = 0

    if Sinv > x[0]:
        Qmax = math.sqrt(Sinv * Sinv - x[0] * x[0])
    else:
        Qmax = 0
    if x[1] > Qmax:
        x[1] = Qmax
    elif x[1] < -Qmax:
        x[1] = -Qmax

    Pavailable = Pavailable + Pmax
    Qavailable = Qavailable + Qmax
    return [x, Pavailable, Qavailable]


def dual_update(mu, coeff_mu, constraint):
    mu_new = mu + coeff_mu * constraint
    mu_new = project_dualvariable(mu_new)
    return mu_new


def matrix_cal_for_subPower(V0, Y00, Y01, Y11, V1_noload):
    diag_V0 = np.matrix([[complex(0, 0)] * 3] * 3)
    diag_V0[0, 0] = V0[0]
    diag_V0[1, 1] = V0[1]
    diag_V0[2, 2] = V0[2]
    K = diag_V0 * Y01.conj() * np.linalg.inv(Y11.conj())
    g = diag_V0 * Y00.conj() * np.matrix(V0).transpose().conj() + diag_V0 * Y01.conj() * V1_noload.conj()

    return [K, g]


def subPower_PQ(V1, PQ_node, K, g):
    diag_V1 = np.matrix([[complex(0, 0)] * len(V1)] * len(V1))
    for ii in range(len(V1)):
        diag_V1[ii, ii] = V1[ii]
    M = K * np.linalg.inv(diag_V1)
    MR = M.real
    MI = M.imag
    P0 = g.real + (MR.dot(PQ_node.real) * 1000 - MI.dot(PQ_node.imag) * 1000)
    Q0 = g.imag + (MR.dot(PQ_node.imag) * 1000 + MI.dot(PQ_node.real) * 1000)

    P0 = P0 / 1000
    Q0 = Q0 / 1000  # convert to kW/kVar

    return [P0, Q0, M]


def sub_costFun_gradient(x, sub_ref, coeff_sub, sub_measure, M, node_withPV):
    grad_a = np.matrix([0] * len(x)).transpose()
    grad_b = np.matrix([0] * len(x)).transpose()
    grad_c = np.matrix([0] * len(x)).transpose()

    MR = M.real
    MI = M.imag
    count = 0
    for node in node_withPV:
        grad_a[count] = -MR[0, int(node)]
        grad_b[count] = -MR[1, int(node)]
        grad_c[count] = -MR[2, int(node)]

        grad_a[count + len(node_withPV)] = MI[0, int(node)]
        grad_b[count + len(node_withPV)] = MI[1, int(node)]
        grad_c[count + len(node_withPV)] = MI[2, int(node)]

        count = count + 1

    res = coeff_sub * ((sub_measure[0] - sub_ref[0]) * 1000 * grad_a + (sub_measure[1] - sub_ref[1]) * 1000 * grad_b
                       + (sub_measure[2] - sub_ref[2]) * 1000 * grad_c)
    res = res / 1000

    return res


def projection(x, xmax, xmin):
    for ii in range(len(x)):
        if x.item(ii) > xmax[ii]:
            x[ii] = xmax[ii]
        if x.item(ii) < xmin[ii]:
            x[ii] = xmin[ii]
    return x


class DERMS:
    def __init__(self, control_node, measure_node, baseload_control_object, PV_control_object, storage_control_object,
                 EV_control_object, WH_control_object, HVAC_control_object, VPP_control_flag, load_control_flag, VR_control_flag):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PV_control_object: information of all controlled PVs
        # storage_control_object: information of all controlled storages
        # WH_control_object: information of all controlled WHs
        # HVAC_control_object: information of all controlled HVACs
        # control_node: node names of all measurements (=control points)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.control_node = control_node
        self.measure_node = measure_node
        self.num_DER_controller = len(control_node)
        self.num_measure_node = len(measure_node)
        self.PV_control_object = PV_control_object
        self.storage_control_object = storage_control_object
        self.EV_control_object = EV_control_object
        self.WH_control_object = WH_control_object
        self.HVAC_control_object = HVAC_control_object
        self.baseload_control_object = baseload_control_object
        self.VPP_control_flag = VPP_control_flag
        self.VR_control_flag = VR_control_flag
        self.load_control_flag = load_control_flag

    def register(self):
        control_object = []
        for nd in self.control_node:
            ctrl = {}
            ctrl["node"] = nd
            ctrl["pv"] = 'nan'
            ctrl["storage"] = 'nan'
            ctrl["EV"] = 'nan'
            ctrl["hvac"] = 'nan'
            ctrl["wh"] = 'nan'
            ctrl["base_load"] = 'nan'
            ctrl["ctrl_load"] = 'nan'
            for pv in self.PV_control_object:
                if pv["bus"].upper() == nd.upper():
                    ctrl["pv"] = pv
            for storage in self.storage_control_object:
                if storage["bus"].upper() == nd.upper():
                    ctrl["storage"] = storage
            for ev in self.EV_control_object:
                if ev["bus"].upper() == nd.upper():
                    ctrl["EV"] = ev
            for hvac in self.HVAC_control_object:
                if hvac["bus"].upper() == nd.upper():
                    ctrl["hvac"] = hvac
            for wh in self.WH_control_object:
                if wh["bus"].upper() == nd.upper():
                    ctrl["wh"] = wh
            for ld in self.baseload_control_object:
                if ld["bus"].upper() == nd.upper():
                    ctrl["base_load"] = ld
                    if self.load_control_flag == 1:
                        ctrl["ctrl_load"] = ld
            control_object.append(ctrl)
        return control_object

    def monitor(self, circuit, dss, feederhead_linename):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # SCADA Voltage and Current at measurement nodes
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Vmes = []
        Imes = []
        Pmes = []
        for node in self.measure_node:
            Vmes.append(dss_function.get_Vnode(dss, circuit, node))

        # <-------- Revise Needed: find the lines connected to measure_node, and get current values. Current values should be in % by getting both actual and limit at the element.
        # controlelem = self.control_elem
        # for elem in controlelem:
        #     current = dss_function.getElemCurrents(circuit, dss, elem.split('.')[0], elem.split('.')[1])
        #     Imes.append(current)
        circuit.SetActiveElement(feederhead_linename)
        Pmes = dss.CktElement.Powers()[0:6:2]
        return [Vmes, Imes, Pmes]

    def coordinator(self, mu0, Vmes, Vupper, Vlower, linear_PFmodel_coeff, AllNodeNames, Pmes, Pset):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # mu0 is the dual variable from last time step: mu_Vmag_upper0, mu_Vmag_lower0
        # Vmes is the list of pu voltages at measurement points, same order as measure_node
        # Vupper and Vlower are user-defined values, could be simply 1.05 and 0.95
        # linear_PF_coeff is the linear power flow model coefficients for the zone, and linear power flow model
        #                coefficients are the result vector from function "linear_powerflow_model"
        # AllNodeNames is a list that includes the names of all nodes in the feeder
        # measure_node, control_node are the lists that include the names of all measurement points, and all nodes with controlled DERs, respectively
        # Pmes is three-phase active power measurement at feeder head
        # Pset is three-phase active power setpoint
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 0. get list of measure nodes and control nodes
        measure_node_index = []
        Vmes_index = []
        for vidx, nd in enumerate(self.measure_node):
            idx = AllNodeNames.index(nd.upper())
            if Vmes[vidx] != 0:
                Vmes_index.append(vidx)
                measure_node_index.append(idx)
        control_node_index = []
        for nd in self.control_node:
            idx = AllNodeNames.index(nd.upper())
            control_node_index.append(idx)

        # 1. update dual variables
        stepsize_mu_upper = 100
        stepsize_mu_lower = 100
        mu_Vmag_upper0 = mu0[0, Vmes_index]
        mu_Vmag_lower0 = mu0[1, Vmes_index]

        if isinstance(Vlower, float):
            Vlower = np.array([Vlower] * len(Vmes))[Vmes_index]
        if isinstance(Vupper, float):
            Vupper = np.array([Vupper] * len(Vmes))[Vmes_index]
        mu_Vmag_lower1 = mu_Vmag_lower0 + stepsize_mu_lower * (Vlower - np.array(Vmes)[Vmes_index])
        mu_Vmag_upper1 = mu_Vmag_upper0 + stepsize_mu_upper * (np.array(Vmes)[Vmes_index] - Vupper)
        mu_Vmag_lower1 = project_dualvariable(mu_Vmag_lower1)
        mu_Vmag_upper1 = project_dualvariable(mu_Vmag_upper1)

        mu1 = mu0
        mu1[0, Vmes_index] = mu_Vmag_upper1
        mu1[1, Vmes_index] = mu_Vmag_lower1

        # 2. compute gradients
        coeff_Vmag_P = linear_PFmodel_coeff[3][
            np.ix_([ii for ii in measure_node_index], [ii for ii in control_node_index])]
        coeff_Vmag_Q = linear_PFmodel_coeff[4][
            np.ix_([ii for ii in measure_node_index], [ii for ii in control_node_index])]
        # coeff_I_P = linear_PFmodel_coeff[6]
        # coeff_I_Q = linear_PFmodel_coeff[7]

        Vmag_upper_gradient_P = np.dot(coeff_Vmag_P.transpose(), mu_Vmag_upper1)
        Vmag_upper_gradient_Q = np.dot(coeff_Vmag_Q.transpose(), mu_Vmag_upper1)
        Vmag_lower_gradient_P = np.dot(coeff_Vmag_P.transpose(), mu_Vmag_lower1)
        Vmag_lower_gradient_Q = np.dot(coeff_Vmag_Q.transpose(), mu_Vmag_lower1)

        if self.VR_control_flag == 0:
            Vmag_upper_gradient_P = np.zeros(len(Vmag_upper_gradient_P))
            Vmag_upper_gradient_Q = np.zeros(len(Vmag_upper_gradient_Q))
            Vmag_lower_gradient_P = np.zeros(len(Vmag_lower_gradient_P))
            Vmag_lower_gradient_Q = np.zeros(len(Vmag_lower_gradient_Q))

        # 3. compute gradient of feeder head power
        if self.VPP_control_flag == 1:
            coeff_Psub_P = linear_PFmodel_coeff[9]
            coeff_Psub_Q = linear_PFmodel_coeff[10]
            Psub_gradient_P = 2 * np.dot(coeff_Psub_P.transpose(),
                                         (np.array(Pmes) - np.array(Pset)) / np.linalg.norm(Pmes))
            Psub_gradient_Q = 2 * np.dot(coeff_Psub_Q.transpose(),
                                         (np.array(Pmes) - np.array(Pset)) / np.linalg.norm(Pmes))
            Psub_gradient_P = Psub_gradient_P[control_node_index]
            Psub_gradient_Q = Psub_gradient_Q[control_node_index]
        else:
            Psub_gradient_P = np.zeros(len(Vmes))
            Psub_gradient_Q = np.zeros(len(Vmes))

        return [mu1, Vmag_upper_gradient_P, Vmag_upper_gradient_Q, Vmag_lower_gradient_P, Vmag_lower_gradient_Q,
                Psub_gradient_P, Psub_gradient_Q]


class DER_controller:
    def __init__(self, control_node_information, coeff_der, coeff_bill, coeff_comf, coeff_sub, battery_SOC_pref):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # coeff_cost=[coeff_pv,coeff_storage,coeff_hvac,coeff_wh]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.ctrl_node = control_node_information["node"]
        self.ctrl_PV = control_node_information["pv"]
        self.ctrl_storage = control_node_information["storage"]
        self.ctrl_EV = control_node_information["EV"]
        self.ctrl_HVAC = control_node_information["hvac"]
        self.ctrl_WH = control_node_information["wh"]
        self.base_load = control_node_information["base_load"]
        self.ctrl_load = control_node_information["ctrl_load"]
        self.coeff_der = coeff_der  # coefficients for individual DERs
        self.coeff_bill = coeff_bill  # bill coefficient
        self.coeff_comf = coeff_comf  # comf coefficient
        self.coeff_sub = coeff_sub
        self.battery_SOC_pref = battery_SOC_pref

    def monitor(self, dss, circuit):
        M = 0
        if self.ctrl_PV != 'nan':
            circuit.SetActiveElement(self.ctrl_PV["name"])
            pv_power = dss.CktElement.Powers()[:2]
            pv_power = [-ii for ii in pv_power]
            M = float(self.ctrl_PV["kW"]) + M
        else:
            pv_power = [0, 0]
        if self.ctrl_storage != 'nan':
            dss.Circuit.UpdateStorage()
            circuit.SetActiveElement(self.ctrl_storage["name"])
            idx = list(dss.utils.class_to_dataframe('Storage').axes[0]).index(self.ctrl_storage["name"])
            temp = float(dss.utils.class_to_dataframe('Storage')['kW'][idx])  # posistive means discharge
            soc = float(dss.utils.class_to_dataframe('Storage')['%stored'][idx])
            storage_power = [temp, soc]
            # circuit.SetActiveElement(self.ctrl_storage["name"])
            # storage_power = dss.CktElement.Powers()[:2]
            # if storage_power[0] > 0: #charge

            #     soc = self.ctrl_storage["SOC0"] + (t-self.ctrl_storage["t0"])*(self.ctrl_storage["efficiency_charge"]*chargekW/self.capacity)- duration*(dischargekW/self.efficiency_discharge/self.capacity)

            M = float(self.ctrl_storage["kWrated"]) + M
        else:
            storage_power = [0, 0]
        if self.ctrl_HVAC != 'nan':
            circuit.SetActiveElement(self.ctrl_HVAC["name"])
            HVAC_power = dss.CktElement.Powers()
        else:
            HVAC_power = [0, 0]
        if self.ctrl_WH != 'nan':
            circuit.SetActiveElement(self.ctrl_WH["name"])
            WH_power = dss.CktElement.Powers()
        else:
            WH_power = [0, 0]
        if self.ctrl_load != 'nan':
            circuit.SetActiveElement(self.ctrl_load["name"])
            load_power = dss.CktElement.Powers()[:2]
        else:
            load_power = [0, 0]
        total_house_power = pv_power[0] + storage_power[0] + HVAC_power[0] + WH_power[0] + load_power[0] + M
        DER_status = [pv_power, storage_power, HVAC_power, WH_power, load_power]

        phase = int(self.ctrl_node.split('.')[1])
        return [DER_status, total_house_power, phase]

    def DER_optimizer(self, DER_status_list, total_house_power_list, PV_Pmax_forecast, PV_Sinv,
                      gradient_from_coordinator, TOU_tariff, dt, load_forecast):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # gradient_from_coordinator is the row vector received from coordinator
        # DER_status and total_house_power should be vectors including multi-time slot resuts
        # dt is the time interval between current control time and last control time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        stepsize_xPpv = 50
        stepsize_xQpv = 50
        stepsize_xstorage = 100
        stepsize_xPload = 500
        stepsize_mu_SOC = 50
        SOC_upper = 80
        SOC_lower = 20
        SOCev_upper = 80
        SOCev_lower = 20
        max_load_curtail = 0.2  # maximum curtailment (*100=%)

        # pv_power_list = []
        # storage_power_list = []
        # hvac_power_list = []
        # wh_power_list = []
        # for ii in range(len(DER_status_list)):
        #     pv_power_list.append(DER_status_list[ii][0])
        #     storage_power_list.append(DER_status_list[ii][1])
        #     hvac_power_list.append(DER_status_list[ii][2])
        #     wh_power_list.append(DER_status_list[ii][3])

        pv_power_list = DER_status_list[0]
        storage_power_list = DER_status_list[1]
        load_power_list = DER_status_list[2]
        EV_power_list = DER_status_list[3]
        # hvac_power_list = DER_status_list[2]
        # wh_power_list = DER_status_list[3]

        # if storage_power_list[1]<=80:
        #     print(1)

        if self.ctrl_PV != 'nan':
            dC_dPpv = -2 * self.coeff_bill * total_house_power_list * TOU_tariff - 2 * self.coeff_comf * self.coeff_der[
                0] * (PV_Pmax_forecast - pv_power_list[0])
            dC_dQpv = 2 * self.coeff_comf * self.coeff_der[1] * pv_power_list[1]
            dCsub_dPpv = self.coeff_sub * gradient_from_coordinator[4]
            dCsub_dQpv = 0  # self.coeff_sub * gradient_from_coordinator[5]
            d_Vmag_dPpv = gradient_from_coordinator[0] - gradient_from_coordinator[2]
            d_Vmag_dQpv = gradient_from_coordinator[1] - gradient_from_coordinator[3]

            # compute x1
            x1_Ppv = pv_power_list[0] - stepsize_xPpv * (dC_dPpv + d_Vmag_dPpv + dCsub_dPpv)
            x1_Qpv = pv_power_list[1] - stepsize_xQpv * (dC_dQpv + d_Vmag_dQpv + dCsub_dQpv)
            [x1, Pmax_allPV, Qmax_allPV] = project_PV([x1_Ppv, x1_Qpv], PV_Pmax_forecast, PV_Sinv)
            x1_Ppv = x1[0]
            x1_Qpv = x1[1]
        else:
            x1_Ppv = 'nan'
            x1_Qpv = 'nan'

        if self.ctrl_load != 'nan':
            dC_dPload = 2 * self.coeff_bill * total_house_power_list * TOU_tariff
            dCsub_dPload = -self.coeff_sub * gradient_from_coordinator[4]
            d_Vmag_dPload = -gradient_from_coordinator[0] + gradient_from_coordinator[2]
            x1_Pload = load_power_list[0] - stepsize_xPload * (dC_dPload + d_Vmag_dPload + dCsub_dPload)
            if x1_Pload > load_forecast:
                x1_Pload = load_forecast
            elif x1_Pload < (1 - max_load_curtail) * load_forecast:
                x1_Pload = (1 - max_load_curtail) * load_forecast
        else:
            x1_Pload = 'nan'

        if self.ctrl_storage != 'nan':
            mu_SOC_upper = self.ctrl_storage["mu_SOC_upper"] + stepsize_mu_SOC * (storage_power_list[1] - SOC_upper)
            mu_SOC_lower = self.ctrl_storage["mu_SOC_lower"] + stepsize_mu_SOC * (SOC_lower - storage_power_list[1])
            mu_SOC_upper = project_dualvariable([mu_SOC_upper])[0]
            mu_SOC_lower = project_dualvariable([mu_SOC_lower])[0]
            self.ctrl_storage["mu_SOC_upper"] = mu_SOC_upper
            self.ctrl_storage["mu_SOC_lower"] = mu_SOC_lower
            dC_dPstorage = -2 * self.coeff_bill * total_house_power_list * TOU_tariff - 2 * self.coeff_comf * \
                           self.coeff_der[2] * (storage_power_list[1] - self.battery_SOC_pref) / 100 * dt * (
                                   self.ctrl_storage['efficiency_charge'] / 100) / self.ctrl_storage['kWhrated']
            d_Vmag_dPstorage = 0  # gradient_from_coordinator[0] - gradient_from_coordinator[2]
            dCsub_dPstorage = -self.coeff_sub * gradient_from_coordinator[4]
            dmuSOC_dPstorage = -mu_SOC_upper * dt * (self.ctrl_storage['efficiency_charge'] / 100) / self.ctrl_storage[
                'kWhrated'] + mu_SOC_lower * dt * (self.ctrl_storage['efficiency_charge'] / 100) / self.ctrl_storage[
                                   'kWhrated']

            x1_Pstorage = storage_power_list[0] - stepsize_xstorage * (
                    dC_dPstorage + d_Vmag_dPstorage + dCsub_dPstorage + dmuSOC_dPstorage)
            if x1_Pstorage <= -self.ctrl_storage["kWrated"]:
                x1_Pstorage = -self.ctrl_storage["kWrated"]
            if x1_Pstorage >= self.ctrl_storage["kWrated"]:
                x1_Pstorage = self.ctrl_storage["kWrated"]
        else:
            x1_Pstorage = 'nan'

        if self.ctrl_EV != 'nan':
            mu_SOCev_upper = self.ctrl_EV["mu_SOC_upper"] + stepsize_mu_SOC * (EV_power_list[1] - SOCev_upper)
            mu_SOCev_lower = self.ctrl_EV["mu_SOC_lower"] + stepsize_mu_SOC * (SOCev_lower - EV_power_list[1])
            mu_SOCev_upper = project_dualvariable([mu_SOCev_upper])[0]
            mu_SOCev_lower = project_dualvariable([mu_SOCev_lower])[0]
            self.ctrl_EV["mu_SOC_upper"] = mu_SOCev_upper
            self.ctrl_EV["mu_SOC_lower"] = mu_SOCev_lower
            dC_dPev = -2 * self.coeff_bill * total_house_power_list * TOU_tariff

            # d_Vmag_dPstorage = gradient_from_coordinator[0] - gradient_from_coordinator[2]
            dCsub_dPev = -self.coeff_sub * gradient_from_coordinator[4]
            dmuSOC_dPev = -mu_SOCev_upper * dt * (self.ctrl_EV['efficiency_charge'] / 100) / self.ctrl_EV[
                'kWhrated'] + mu_SOCev_lower * dt * (self.ctrl_EV['efficiency_charge'] / 100) / self.ctrl_EV['kWhrated']
            x1_Pev = EV_power_list[0] - stepsize_xstorage * (
                    dC_dPev + dmuSOC_dPev + dCsub_dPev)  # + 100*d_Vmag_dPstorage # solved based on discharge (Pdischarge-Pcharge)
            if x1_Pev <= -self.ctrl_EV["kWrated"]:
                x1_Pev = -self.ctrl_EV["kWrated"]
            if x1_Pev >= 0:
                x1_Pev = 0

        else:
            x1_Pev = 'nan'

        return [x1_Ppv, x1_Qpv, x1_Pstorage, x1_Pload, x1_Pev]


def project_Storage(Pcharge, Pdischarge, charge_rate, discharge_rate, charge_lower, discharge_lower):
    if Pcharge > charge_rate:
        Pcharge = charge_rate
    elif Pcharge < charge_lower:
        Pcharge = charge_lower
    if Pdischarge > discharge_rate:
        Pdischarge = discharge_rate
    elif Pdischarge < discharge_lower:
        Pdischarge = discharge_lower
    return [Pcharge, Pdischarge]


if __name__ == "__main__":
    import os
    import json

    fname = os.path.join(os.path.dirname(__file__), 'Heila Coordinator Data.txt')
    assert os.path.exists(fname)
    with open(fname, 'r') as f:
        data = json.load(f)
    input = data['input']
    output = data['output']
    print(input.keys())
