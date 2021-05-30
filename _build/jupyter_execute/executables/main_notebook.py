#!/usr/bin/env python
# coding: utf-8

# # Welcome to Q³ calculator

# In[1]:


# CODE BEGINS

import ipywidgets as widgets
import pyscf, qiskit, qiskit_nature
import pathlib
from pyscf.geomopt.geometric_solver import optimize as geomoptimize
from pyscf.geomopt.berny_solver import optimize as bernyoptimize
import ipywidgets as widgets

# Setting some defaults
molecule_file = None # input file to look for
default_selection_width = "60%"
style = {'description_width': 'initial'}
hbegin = "<"+"h2"+">" # using '+' to avoid HTML look in an unrendered ipynb
hend = "<"+"/h2"+">"

# Some geometries

stored_geometries = {"Hydrogen": [['H',  0.00,  0.00,  0.00],
                                  ['H',  0.76,  0.00,  0.00]],
                     "Water":    [['O',  0.00,  0.00,  0.00],
                                  ['H',  0.28,  0.89,  0.25],
                                  ['H',  0.61, -0.24, -0.72]],
                     "Ethene":   [['C', -0.6672,  0.0000,  0.0000],
                                  ['C',  0.6672,  0.0000,  0.0000],
                                  ['H', -1.2213, -0.9290,  0.0708],
                                  ['H', -1.2212,  0.9290, -0.0708],
                                  ['H',  1.2213,  0.9290, -0.0708],
                                  ['H',  1.2213, -0.9290,  0.0708]]
                    }
default_geometry = "Hydrogen"

# Some convergence parameters

global conv_params # this dictionary is later stripped down to hold only the chosen parameters
conv_params = {"default": { # These are the default settings
    'convergence_energy': 1e-6,  # Eh
    'convergence_grms': 3e-4,    # Eh/Bohr
    'convergence_gmax': 4.5e-4,  # Eh/Bohr
    'convergence_drms': 1.2e-3,  # Angstrom
    'convergence_dmax': 1.8e-3,  # Angstrom
},               "tight": {
    'convergence_energy': 1e-8,  # Eh
    'convergence_grms': 2e-4,    # Eh/Bohr
    'convergence_gmax': 2.5e-4,  # Eh/Bohr
    'convergence_drms': 0.5e-3,  # Angstrom
    'convergence_dmax': 0.5e-3,  # Angstrom
},              "custom": {
    'convergence_energy': 1e-8,  # Eh
    'convergence_grms': 2e-4,    # Eh/Bohr
    'convergence_gmax': 2.5e-4,  # Eh/Bohr
    'convergence_drms': 0.5e-3,  # Angstrom
    'convergence_dmax': 0.5e-3,  # Angstrom
}} 

# Setup a molecule
global mol
mol = pyscf.gto.Mole()
mol.atom = stored_geometries[default_geometry]
mol.unit = 'A'
mol.verbose = 0

# Starting with UI
# All widgets logic
# For step 1
titlebar = widgets.HTML(value=hbegin.replace('2','1')+"Qiskit QuESt Calculator"+hend.replace('2','1'))

subtitle1 = widgets.HTML(value=hbegin+"Select a molecule"+hend)

file_picker = widgets.Dropdown(options= list(stored_geometries.keys())+['Upload own file'],
                               value = default_geometry, description="Pick a molecule",
                               style=style, layout=widgets.Layout(width=default_selection_width))

file_upload = widgets.FileUpload(accept='.xyz,.mol', multiple=False, description="Select File",
                                 disabled=True, layout=widgets.Layout(width=default_selection_width))

file_box = widgets.HBox(children=[file_picker, file_upload],
                        layout=widgets.Layout(width=default_selection_width))

confirm_file_button = widgets.Button(description="Use "+default_geometry+" molecule",
                                     disabled=False, layout=widgets.Layout(width=default_selection_width))

# For Step 2
subtitle2 = widgets.HTML(value=hbegin+"Do Classical Calculation"+hend)

select_basis = widgets.Dropdown(options=['sto-3g', 'sto-6g', 'cc_pVTZ', 'def2-tzvp'], disabled=True,
                                value = 'sto-3g', description="Basis set for the atoms",
                                style=style, layout=widgets.Layout(width=default_selection_width))
select_spin = widgets.Dropdown(options=[("spin 0 / multiplicity singlet",0),
                                        ("spin 1 / multiplicity doublet",1),
                                        ("spin 2 / multiplicity triplet",2)], disabled=True,
                               value = 0, description="Spin (number of unparied electrons)",
                               style=style, layout=widgets.Layout(width=default_selection_width))
select_symmetry = widgets.ToggleButtons(options=[True,False], disabled=True,
                                        value = True, description="Make use of Point Group Symmetry",
                                        style=style, layout=widgets.Layout(width=default_selection_width))
select_charge = widgets.SelectionSlider(options=[-1,0,1], disabled=True,
                                        value = 0, description="Charge on the molecule",
                                        style=style, layout=widgets.Layout(width=default_selection_width))



select_method = widgets.Dropdown(options=[('Hartree-Fock (HF)', 'HF'),('Kohn-Sham (KS)','KS')], disabled=True,
                                 value = 'HF', description="Method", style=style,
                                 layout=widgets.Layout(width=default_selection_width))
select_geooptimizer = widgets.Dropdown(options=[('geomeTRIC','geometric'),('PyBerny', 'pyberny')], disabled=True,
                                       value = 'geometric', description="Geometry Optimizer",
                                       style=style, layout=widgets.Layout(width=default_selection_width))

criteria_description = {'convergence_energy':"Energy (hartree)",
                        'convergence_grms': " XYZABC (hartree/bohr)",
                        'convergence_gmax': " XYZABC (hartree/bohr)",
                        'convergence_drms': " XYZABC (Angstrom)",
                        'convergence_dmax': " XYZABC (Angstrom)",
                       }

default_settings = []
for criteria in conv_params["default"].keys():
    default_settings.append(widgets.FloatText(value=conv_params["default"][criteria], description =  criteria_description[criteria],
                                            style=style, disabled=True, layout=widgets.Layout(width="99%")))
default_settings = widgets.VBox(children=default_settings)

tight_settings = []
for criteria in conv_params["tight"].keys():
    tight_settings.append(widgets.FloatText(value=conv_params["tight"][criteria], description =  criteria_description[criteria],
                                            style=style, disabled=True, layout=widgets.Layout(width="99%")))
tight_settings = widgets.VBox(children=tight_settings)

get_custom_settings = []
for criteria in conv_params["default"].keys():
    get_custom_settings.append(widgets.FloatText(value=conv_params["default"][criteria], description =  criteria_description[criteria],
                                            style=style, disabled=True, layout=widgets.Layout(width="99%")))
custom_input = widgets.VBox(children=get_custom_settings)

select_conv_params = widgets.Tab(children = [default_settings, tight_settings, custom_input], disabled=True,
                                 layout=widgets.Layout(width=default_selection_width))
select_conv_params._titles = {0:"Default criteria",1:"Tight criteria", 2:"Custom criteria"}

confirm_classical_settings = widgets.Button(description="Run Classical Calculation",
                                  disabled=True, layout=widgets.Layout(width=default_selection_width))

classical_result = widgets.Label(value="")


# All button switching logic

# For Step 1

def file_picker_switch(value): # Only allow file upload if option is chosen
    if value['new'] in stored_geometries:
        global mol
        mol.atom = stored_geometries[value['new']]
        file_upload.disabled = True
        confirm_file_button.description = "Use "+value['new']+" molecule"
        confirm_file_button.disabled = False
    elif value['new'] == 'Upload own file':
        file_upload.disabled = False
        if file_upload.value: # in case a file was picked previously
            confirm_file_button.description = "Upload "+file_upload.metadata[0]['name']
        else:
            confirm_file_button.description = "Pick an .xyz or .mol file"
            confirm_file_button.disabled = True

def upload_button_used(value): # Rename upload button to show picked filename
    if file_upload.value:
        if file_upload.metadata[0]['name'][-3:] in ["xyz","mol"]:
            file_upload.description = "Upload "+file_upload.metadata[0]['name']
            confirm_file_button.description = "Use my file: "+file_upload.metadata[0]['name']
            confirm_file_button.disabled = False
        else:
            confirm_file_button.description = "Please pick an .xyz or .mol file"

def fileread_failed():
    print("Error understanding uploaded file. Use another.")
    print("Reload page to ensure fresh start!")

def convert_mol_to_xyz(lines):
    try:
        number_of_atoms = int(lines[3].split()[0]) # num of atoms are mentioned on line 4
        xyz = []
        for line in lines[4:number_of_atoms+4]:
            data = line.split()
            xyz.append([data[3], float(data[0]), float(data[1]), float(data[2])])
    except:
        fileread_failed()
    return xyz
        
def read_file(filename):
    kind = filename[-3:]
    content = file_upload.data[0].decode()
    content = content.splitlines()
    if kind == "mol":
        content, kind = convert_mol_to_xyz(content), "xyz"
    return content

def start_step_2():
    file_picker.unobserve_all()
    file_picker.disabled = True
    file_upload.unobserve_all()
    file_upload.disabled = True
    confirm_file_button.unobserve_all()
    confirm_file_button.disabled = True
    select_basis.disabled = False
    select_spin.disabled = False
    select_symmetry.disabled = False
    select_charge.disabled = False
    select_method.disabled = False
    select_geooptimizer.disabled = False
    for widget in get_custom_settings:
        widget.disabled = False
    confirm_classical_settings.disabled = False

def file_confirmed(_):
    if not file_upload.disabled: # i.e. a file was uploaded
        molecule_file = (file_upload.description).lstrip("Upload ")
        assert molecule_file == file_upload.metadata[0]['name']
        global mol
        mol.atom = read_file(molecule_file)
        try: # to check validity of uploaded geometry
            temp_mol = mol.copy() 
            temp_mol.basis = 'sto-3g' 
            temp_mol.build()
            del temp_mol
        except:
            fileread_failed()
    else:
        molecule_file = str(file_picker.value)+".xyz"
    start_step_2()
    return molecule_file

def start_step_3():
    select_basis.disabled = True
    select_spin.disabled = True
    select_symmetry.disabled = True
    select_charge.disabled = True
    select_method.disabled = True
    select_geooptimizer.disabled = True
    for widget in get_custom_settings:
        widget.disabled = True
    confirm_classical_settings.disabled = True
    
    

def classical_settings_confirmed(_):
    start_step_3()
    global mol
    global conv_params
    mol.basis = select_basis.value
    mol.spin  = select_spin.value
    mol.symmetry = select_symmetry.value
    mol.charge = select_charge.value
    mol.build()
    if select_conv_params.selected_index == 0:
        conv_params = conv_params["default"]
    elif select_conv_params.selected_index == 1:
        conv_params = conv_params["tight"]
    elif select_conv_params.selected_index == 2:
        for i, criteria in enumerate(conv_params["custom"].keys()):
            conv_params["custom"][criteria] = get_convergence_widgets[i].value
        conv_params = conv_params["custom"]
    if select_method.value == 'HF':
        mf = pyscf.scf.HF(mol) # mf stands for mean-field
        # the mean-field object stores final converged ground-state energy, MO coefficients, occupations, etc. 
    elif select_method.value == 'KS':
        mf = pyscf.scf.KS(mol)
        mf.xc = "pbe"
    mf.kernel() # setup kernel
    mol = geomoptimize(mf, **conv_params)
    classical_result.value = "The classically calculated ground state energy of the system is: "+str(mf.e_tot)+" hartree."

# For Step 2


# Observers
    
file_picker.observe(file_picker_switch, names='value') # Monitor option chosen by file picker
file_upload.observe(upload_button_used, names = 'value') # Monitor which file was picked
molecule_file = confirm_file_button.on_click(file_confirmed) # To move to step 2
confirm_classical_settings.on_click(classical_settings_confirmed) # To move to step 3



# Build the full widget
calculator = widgets.VBox(children=[titlebar,subtitle1,
                                    file_box,confirm_file_button,
                                    subtitle2,
                                    select_basis, select_spin, select_symmetry,select_charge,
                                    select_method, select_geooptimizer,
                                    widgets.Label(value="$Convergence$ $settings $"),
                                    select_conv_params, confirm_classical_settings,
                                    classical_result])


# In[2]:


display(calculator) # Please wait, the calculator will appear here

