#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from pymatgen import Element
fe=Element("Fe")
#   for x in dir(fe):
#       print(x)
#   for k, v in fe.data.items():
#       print(k,v)
print(fe.data)
print(fe.Z)
print(fe.symbol)
print(fe.X)
print(fe.number)
print(fe.max_oxidation_state)
print(fe.min_oxidation_state)
print(fe.oxidation_states)
print(fe.common_oxidation_states)
print(fe.full_electronic_structure)
print(fe.row)
print(fe.group)
print(fe.block)
print(fe.is_noble_gas)
print(fe.is_transition_metal)
print(fe.is_rare_earth_metal)
print(fe.is_metalloid)
print(fe.is_alkali)
print(fe.is_alkaline)
print(fe.is_halogen)
print(fe.is_chalcogen)
print(fe.is_lanthanoid)
print(fe.is_actinoid)
print(fe.long_name)                               # html
print(fe.atomic_mass)                             # html
print(fe.atomic_radius)                           # html
print(fe.van_der_waals_radius)                    # html
print(fe.mendeleev_no)                            # html
print(fe.electrical_resistivity)                  # html
print(fe.velocity_of_sound)                       # html
print(fe.reflectivity)                            # html
print(fe.refractive_index)                        # html
print(fe.poissons_ratio)                          # html
print(fe.molar_volume)                            # html
print(fe.electronic_structure)                    # html
print(fe.atomic_orbitals)                         # html
print(fe.thermal_conductivity)                    # html
print(fe.boiling_point)                           # html
print(fe.melting_point)                           # html
print(fe.critical_temperature)                    # html
print(fe.superconduction_temperature)             # html
print(fe.liquid_range)                            # html
print(fe.bulk_modulus)                            # html
print(fe.youngs_modulus)                          # html
print(fe.brinell_hardness)                        # html
print(fe.rigidity_modulus)                        # html
print(fe.mineral_hardness)                        # html
print(fe.vickers_hardness)                        # html
print(fe.density_of_solid)                        # html
print(fe.coefficient_of_linear_thermal_expansion) # html
print(fe.average_ionic_radius) 
print(fe.ionic_radii)
print(fe.is_valid_symbol("Fe"))     # dir
print(fe.from_Z(26))                # dir
print(fe.as_dict())                 # dir
print(fe.from_dict(fe.as_dict()))   # dir
print(fe.from_row_and_group(4,8))   # dir
print(fe.ground_state_term_symbol)  # dir
print(fe.icsd_oxidation_states)     # dir
print(fe.name)                      # dir
print(fe.print_periodic_table())    # dir
print(fe.term_symbols)              # dir
print(fe.valence)                   # dir
print(fe.value)                     # dir
print(fe.boiling_point)
print(float(fe.boiling_point.to("")))
