import opendssdirect as dss
import pandas as pd

from core import dss_function

# Author: Michael Blonsky

ELEMENT_CLASSES = {
    'Load': dss.Loads,
    'PV': dss.PVsystems,
    'Generator': dss.Generators,
    'Line': dss.Lines,
    'Xfmr': dss.Transformers,
}
LINE_CLASSES = ['Line', 'Xfmr']


class OpenDSSException(Exception):
    pass

class OpenDSS:
    def __init__(self, dss_file, time_step, start_time, redirects_pre=None, redirects_post=None, **kwargs):
        # Run redirect files before main dss file
        if redirects_pre is not None:
            if not isinstance(redirects_pre, list):
                redirects_pre = [redirects_pre]
            for redirect in redirects_pre:
                self.redirect(redirect)

        print('DSS Compiling...')
        self.run_command("Compile " + dss_file)

        # Run redirect files after main dss file
        if redirects_post is not None:
            if not isinstance(redirects_post, list):
                redirects_post = [redirects_post]
            for redirect in redirects_post:
                self.redirect(redirect)

        print('DSS Compiled Circuit:', dss.Circuit.Name())

        self.run_command('Solve')
        summary = dss.run_command('summary')
        print(summary)

        # Set to QSTS Mode
        self.run_command('set mode=yearly')  # Set to QSTS mode
        # dss.Solution.Mode(2)  # should set mode to yearly?

        dss.Solution.Number(1)  # Number of Monte Carlo simulations
        dss.Solution.Hour(start_time.timetuple().tm_yday * 24 + start_time.hour)  # QSTS starting hour

        # Run, without advancing, then set step size
        dss.Solution.StepSize(0)
        self.run_dss()
        dss.Solution.StepSize(time_step.total_seconds())


        # print('DSS Compiled Circuit:', dss.Circuit.Name())

    @staticmethod
    def run_command(cmd):
        status = dss.run_command(cmd)
        if status:
            print('DSS Status ({}): {}'.format(cmd, status))

    def redirect(self, filename):
        self.run_command('Redirect ' + filename)

    @staticmethod
    def run_dss(no_controls=False):
        if no_controls:
            status = dss.Solution.SolveNoControl()
        else:
            status = dss.Solution.Solve()
        if status:
            print('DSS Solve Status: {}'.format(status))

    # GET METHODS

    @staticmethod
    def get_all_buses():
        return dss.Circuit.AllBusNames()

    @staticmethod
    def get_all_elements(element='Load'):
        if element in ELEMENT_CLASSES:
            cls = ELEMENT_CLASSES[element]
            df = dss.utils.to_dataframe(cls)
        else:
            df = dss.utils.class_to_dataframe(element, transform_string=lambda x: pd.to_numeric(x, errors='ignore'))
            # df = dss.utils.class_to_dataframe(element)
        return df

    @staticmethod
    def get_circuit_power():
        # returns negative of circuit power (positive = consuming power)
        powers = dss.Circuit.TotalPower()
        if len(powers) == 2:
            p, q = tuple(powers)
            return -p, -q
        elif len(powers) == 6:
            p = powers[0:2:]
            q = powers[1:2:]
            return p, q
        else:
            raise OpenDSSException('Expected 1- or 3-phase circuit')

    @staticmethod
    def get_losses():
        p, q = dss.Circuit.Losses()
        return p / 1000, q / 1000

    def get_circuit_info(self):
        # TODO: Add powers by phase if 3-phase; options to add/remove element classes
        p_total, q_total = self.get_circuit_power()
        p_loss, q_loss = self.get_losses()
        p_load, q_load = self.get_total_power()
        p_pv, q_pv = self.get_total_power(element='PV')
        p_gen, q_gen = self.get_total_power(element='Generator')
        p_stor, q_stor = self.get_total_power(element='Storage')
        return {
            'Total P (MW)': p_total / 1000,
            'Total Q (MVAR)': q_total / 1000,
            'Total Loss P (MW)': p_loss / 1000,
            'Total Loss Q (MVAR)': q_loss / 1000,
            'Total Load P (MW)': p_load / 1000,
            'Total Load Q (MVAR)': q_load / 1000,
            'Total PV P (MW)': p_pv / 1000,
            'Total PV Q (MVAR)': q_pv / 1000,
            'Total Generators P (MW)': p_gen / 1000,
            'Total Generators Q (MVAR)': q_gen / 1000,
            'Total Storage P (MW)': p_stor / 1000,
            'Total Storage Q (MVAR)': q_stor / 1000,
        }

    def get_power(self, name, element='Load', phase=None, total=False):
        # If phase=<int>, return length-2 tuple (Pa, Qa) or length-4 tuple (P1a, Q1a, P2a, Q2a), for Lines
        # If phase=None and total=False, return length-2*n_phases, or length-4*n_phases for Lines
        # If phase=None and total=True, return length-2 tuple (Pa+Pb+Pc, Qa+Qb+Qc)
        self.set_element(name, element)
        
        if element == 'PV':
            NumPhase = dss.CktElement.NumPhases()
            powers = dss.CktElement.Powers()[0:2*NumPhase]
        else:
            powers = dss.CktElement.Powers()

        
        
        if phase is None:
            if total:
                if element == 'PV':
                    p = sum(powers[0:len(powers):2])
                    q = sum(powers[1:len(powers):2])
                else:
                    powers = [p for i, p in enumerate(powers) if i % 4 in [0, 1]]
                    p = sum(powers[0::2])
                    q = sum(powers[1::2])
                return p, q
            else:
                if element not in LINE_CLASSES:
                    powers = [p for i, p in enumerate(powers) if i % 4 in [0, 1]]
                return tuple(powers)

        elif phase - 1 in range(len(powers) // 4):
            powers = powers[4 * (phase - 1):4 * phase]
            if element not in LINE_CLASSES:
                powers = powers[:2]
            return tuple(powers)
        else:
            raise OpenDSSException('Bad phase for {} {}: {}'.format(element, name, phase))

    # TODO: alternative to above, not yet tested (need to check for 3 phase loads and lines)
    # def get_p_q(self, name, element='Load', bus=1):
    #     self.set_element(name, element)
    #     cls = ELEMENT_CLASSES[element]
    #     try:
    #         p, q = cls.kW(), cls.kvar()
    #         return p, q
    #     except:
    #         # Need to find the correct exception
    #         return self.get_power(name, element, bus)

    def get_total_power(self, element='Load'):
        p_total, q_total = 0, 0

        if element in ELEMENT_CLASSES:
            cls = ELEMENT_CLASSES[element]
            all_names = cls.AllNames()
            for name in all_names:
                p, q = self.get_power(name, element, total=True)
                p_total += p
                q_total += q
        elif element == 'Storage':
            df_storage = self.get_all_elements('Storage')
            if len(df_storage):
                p_total = df_storage['kW'].sum()
                q_total = df_storage['kvar'].sum()

        return p_total, q_total

    @staticmethod
    def get_bus_voltage(bus, phase=None, pu=True, polar=True, mag_only=True):
        dss.Circuit.SetActiveBus(bus)
        if polar:
            if pu:
                v = dss.Bus.puVmagAngle()
            else:
                v = dss.Bus.VMagAngle()
            if len(v) == 4:  # remove zeros for single phase voltage
                v = v[:2]
            if any([x <= 0 for x in v[::2]]):
                #raise OpenDSSException('Bus "{}" voltage = {}, out of bounds'.format(bus, v))
                v=[0]
            if mag_only:  # remove angles
                v = v[::2]
        else:
            if pu:
                v = dss.Bus.PuVoltage()
            else:
                v = dss.Bus.Voltages()

        if phase is not None and len(v) % 3 == 0:  # 3 phase bus
            l = len(v) // 3
            v = v[l*(phase-1): l*phase]

        if len(v) == 1:
            return v[0]
        else:
            return tuple(v)

    def get_voltage(self, name, element='Load', **kwargs):
        # note: for lines/transformers, always takes voltage from Bus1
        self.set_element(name, element)
        bus = dss.CktElement.BusNames()[0]
        return self.get_bus_voltage(bus, **kwargs)

    def get_all_complex(self, name, element='Load'):
        self.set_element(name, element)
        return {
            'Voltages': dss.CktElement.Voltages(),
            'VoltagesMagAng': dss.CktElement.VoltagesMagAng(),
            'Currents': dss.CktElement.Currents(),
            'CurrentsMagAng': dss.CktElement.CurrentsMagAng(),
            'Powers': dss.CktElement.Powers(),
        }

    # SET METHODS

    @staticmethod
    def set_element(name, element):
        # dss.Circuit.SetActiveElement(self.__Class + '.' + self.__Name)
        name = name.lower()
        cls = ELEMENT_CLASSES[element]
        cls.Name(name)
        if cls.Name() != name:
            raise OpenDSSException('{} "{}" does not exist'.format(element, name))

    def set_power(self, name, p=None, q=None, element='Load', size=None):
        if element in ELEMENT_CLASSES:
            self.set_element(name, element)
            cls = ELEMENT_CLASSES[element]
            if p is not None:
                cls.kW(p)
            if q is not None:
                cls.kvar(q)
        elif element == 'Storage':
            if p > 0:  # charge
                self.run_command('{}.{}.state=charging %charge={}'.format(element, name, abs(p) / size * 100))
            elif p < 0:  # discharge
                self.run_command('{}.{}.state=discharging %discharge={}'.format(element, name, abs(p) / size * 100))
            else:  # idle
                self.run_command('{}.{}.state=idling'.format(element, name))
        else:
            raise OpenDSSException("Unknown element class:", element)

    @staticmethod
    def set_tap(name, tap, max_tap=16):
        dss.RegControls.Name(name)
        tap = min(max(int(tap), -max_tap), max_tap)
        dss.RegControls.TapNumber(tap)

    @staticmethod
    def get_tap(name):
        dss.RegControls.Name(name)
        return int(dss.RegControls.TapNumber())


def save_linear_power_flow(**kwargs):
    AllNodeNames, Vbase_allnode, node_number = dss_function.get_node_information(dss)
    _ = dss_function.system_topology_matrix_form(dss, AllNodeNames, **kwargs)


if __name__ == "__main__":
    from constants import master_dssfile, load_dssfile_all, pv_dssfile, freq_all, start_time

    d = OpenDSS(master_dssfile, freq_all, start_time)
    d.redirect(load_dssfile_all)
    d.redirect(pv_dssfile)
    ##d.redirect(storage_dssfile)

    print('run output:', d.run_dss())
    print()

    # check circuit functions
    print('circuit info:')
    info = d.get_circuit_info()
    for key, val in info.items():
        print(key, val)
    print()

    # All Element Names
    print('DSS Elements: ', dss.Circuit.AllElementNames())

    # All Loads, as dataframe
    df_loads = d.get_all_elements()
    print('First 5 Loads (DataFrame)')
    print(df_loads.head())

    # All Storages, as dataframe
    df_storage = d.get_all_elements('Storage')
    print('First 5 Storages (DataFrame)')
    print(df_storage.head())

    # check bus voltages
    buses = d.get_all_buses()
    bus0 = buses[0]
    print('First Bus voltage:', d.get_bus_voltage(buses[0]))
    print('First Bus voltage, phase1:', d.get_bus_voltage(buses[0], phase=1))
    print('First Bus voltage, complex:', d.get_bus_voltage(buses[0], polar=False, pu=False))
    print('First Bus voltage, MagAng:', d.get_bus_voltage(buses[0], mag_only=False))

    # check load functions
    load_names = d.get_all_elements().index
    print('Load {} data: {}'.format(load_names[0], d.get_all_complex(load_names[0])))
    print('load voltage:', d.get_voltage(load_names[0]))
    print('load powers:', d.get_power(load_names[0]))
    print()

    # checking setting load power
    d.set_power(load_names[0], p=10)

    # check line functions
    # line = 'pc-28179'
    # print('Line powers:', d.get_power(line, element='Line'))
    # print('Line voltages:', d.get_voltage(line, element='Line'))
