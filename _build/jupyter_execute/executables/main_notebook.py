#!/usr/bin/env python
# coding: utf-8

# # Welcome to Q³
# ## Qiskit QuESt Qalculator
# QuESt, or Quantum Electronic Structure, is the new way of doing chemistry!

# #### To launch the application, please hover on the `rocket` icon (top right) and then click on `Live Code`
# Then please wait a while till the backend is ready :)
# ![Instructions](https://raw.githubusercontent.com/QuESt-Calculator/QuESt/main/executables/LiveLaunchScreenshot.png "Instructions")

# In[1]:


# CODE BEGINS

import ipywidgets as widgets
import pyscf, py3Dmol, qiskit, qiskit_nature
import pathlib, io, time
from pyscf.geomopt.geometric_solver import optimize as geomoptimize
from pyscf.geomopt.berny_solver import optimize as bernyoptimize
from IPython.display import FileLink, Image
from IPython.utils.io import capture_output

# Setting some defaults
molecule_file = None # input file to look for
default_selection_width = "60%"
style = {'description_width': 'initial'}
hbegin = "<"+"h2"+">" # using '+' to avoid HTML look in an unrendered ipynb
hend = "<"+"/h2"+">"
isosurface_value = 0.03
opacity = 0.5

# Some geometries

stored_geometries = {"Hydrogen": [['H',  0.00,  0.00,  0.00],
                                  ['H',  0.71,  0.00,  0.00]],
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
global mol, mf, view3D
mol = pyscf.gto.Mole()
mf = None
mol.atom = stored_geometries[default_geometry]
mol.unit = 'A'
mol.output = "geoopt.output"

# Starting with UI

# All widgets logic
# For step 1
titlebar = widgets.HTML(value=hbegin.replace('2','1')+"Qiskit QuESt Calculator"+hend.replace('2','1'))

errorbar = widgets.HTML(value="")
pyscf_out = widgets.Output()

def error_occured(error_message):
    errorbar.value = error_message
    exit()

subtitle1 = widgets.HTML(value=hbegin+"Select a molecule"+hend)

file_picker = widgets.Dropdown(options= list(stored_geometries.keys())+['Upload own file'],
                               value = default_geometry, description="Pick a molecule",
                               style=style, layout=widgets.Layout(width=default_selection_width))

file_upload = widgets.FileUpload(accept='.xyz,.mol', multiple=False, description="Select File",
                                 disabled=True, layout=widgets.Layout(width=default_selection_width))

file_box = widgets.HBox(children=[file_picker, file_upload],
                        layout=widgets.Layout(width=default_selection_width))

confirm_file_button = widgets.Button(description="Use "+default_geometry+" molecule", button_style="success",
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

select_verbosity = widgets.SelectionSlider(options=["Minimal", "Optimal", "Full"], value="Optimal",
                                           description="Verbosity of output file", disabled=True,
                                           style=style, layout=widgets.Layout(width=default_selection_width))

confirm_classical_settings = widgets.Button(description="Run Classical Calculation", button_style="success",
                                  disabled=True, layout=widgets.Layout(width=default_selection_width))

# For step 3

subtitle3 = widgets.HTML(value=hbegin+"Analyze Classical Results"+hend)

classical_energy = widgets.Label(value="")

p3Dw = py3Dmol.view()
select_visual = widgets.ToggleButtons(options=["Geometry","Density","Molecular Orbital"], disabled=True,
                                      value = "Geometry", description="Plot:")
select_mo = widgets.IntSlider(value=1, min=1, max=1, step=1, description="MO number", disabled=True)
        
visualization = widgets.VBox(children=[select_visual, select_mo],
                             layout=widgets.Layout(width=default_selection_width))

classical_result_label = widgets.Label(value="")
classical_result_link = widgets.Output()
download_classical_result = widgets.HBox(children=[classical_result_label, classical_result_link],
                                         layout=widgets.Layout(width=default_selection_width))

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

def convert_to_atom_data(molecule_file):
    if not molecule_file == file_upload.metadata[0]['name']:
        print("You shouldn't get this message") # kind of assertion check
    lines = (file_upload.data[0].decode()).splitlines()
    kind = molecule_file[-3:]
    xyz = []
    mol_line_decompose = lambda data : [data[3], float(data[0]), float(data[1]), float(data[2])]
    xyz_line_decompose = lambda data : [data[0], float(data[1]), float(data[2]), float(data[3])]
    try:
        if kind == "mol":
            # num of atoms are mentioned on line 3, atomic data starts from line 4, counting starts from zeroth line
            number_of_atoms = int(lines[3].split()[0])
            begin_line_number = 4
            line_decompose = mol_line_decompose
        elif kind == "xyz":
            # num of atoms are mentioned on line 0, atomic data starts from line 2, counting starts from zeroth line
            number_of_atoms = int(lines[0].split()[0])
            begin_line_number = 2
            line_decompose = xyz_line_decompose
        end_line_number = begin_line_number+number_of_atoms
        for line in lines[begin_line_number:end_line_number]:
            xyz.append(line_decompose(line.split()))
    except:
        error_occured("Error understanding uploaded file. Use another.<br>Reload page to ensure fresh start!")
    return xyz

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
    select_verbosity.disabled = False
    for widget in get_custom_settings:
        widget.disabled = False
    confirm_classical_settings.disabled = False

def file_confirmed(_):
    global mol
    temp_mol = mol.copy()
    temp_mol.output = "tmp.output"
    temp_mol.basis = 'sto-3g' 
    if not file_upload.disabled: # i.e. a file was uploaded
        molecule_file = ((file_upload.description).lstrip("Upload")).strip()
        if not molecule_file == file_upload.metadata[0]['name']:
            print("You shouldn't get this message") # kind of assertion check
        temp_mol.atom = convert_to_atom_data(molecule_file)
    else:
        molecule_file = str(file_picker.value)+".xyz"
    try: # to check validity of uploaded geometry / included geometry and to generate xyz for 3D view
        temp_mol.build()
        mol.atom = temp_mol.atom # because build was a success
    except:
        error_occured("Error understanding uploaded file. Use another.<br>Reload page to ensure fresh start!")
    temp_mol.tofile("given_molecule.xyz",format="xyz") # for temporary geometry
    p3Dw.removeAllModels()
    p3Dw.removeAllShapes() # just to be sure
    p3Dw.addModel(open("given_molecule.xyz",'r').read(), "xyz")
    p3Dw.setStyle({'stick':{'radius': 0.2}, 'sphere':{'radius': 0.3}})
    p3Dw.update()
    del temp_mol
    start_step_2()

# For Step 2

def start_step_3a():
    select_basis.disabled = True
    select_spin.disabled = True
    select_symmetry.disabled = True
    select_charge.disabled = True
    select_method.disabled = True
    select_geooptimizer.disabled = True
    for widget in get_custom_settings:
        widget.disabled = True
    confirm_classical_settings.disabled = True
    select_verbosity.disabled = True

def start_step_3b():
    select_visual.disabled = False

def visual_switched(value):
    select_mo.disabled = True
    if value['new'] == "Geometry":
        p3Dw.removeAllShapes() # just to be sure
        p3Dw.setStyle({'stick':{'radius': 0.2}, 'sphere':{'radius': 0.3}})
        p3Dw.zoomTo()
        p3Dw.update()
    elif value['new'] == "Density":
        p3Dw.removeAllShapes()
        cube_data = open("electron_density.cube").read()
        p3Dw.addVolumetricData(cube_data, "cube", {'isoval': isosurface_value, 'color': "red", 'opacity': opacity})
        p3Dw.setStyle({'stick':{'radius': 0.1}, 'sphere':{'radius': 0.2}})
        p3Dw.zoomTo()
        p3Dw.update()
    elif value['new'] == "Molecular Orbital":
        select_mo.disabled = False
        mo_changed(None)
    

def mo_changed(_):
    select_mo.description = "Loading..."
    cube_data = open("orb_num_"+str(select_mo.value-1)+".cube").read()
    p3Dw.removeAllShapes()
    p3Dw.addVolumetricData(cube_data, "cube", {'isoval': -isosurface_value, 'color': "red", 'opacity': opacity})
    p3Dw.addVolumetricData(cube_data, "cube", {'isoval':  isosurface_value, 'color': "blue", 'opacity': opacity})
    p3Dw.setStyle({'stick':{'radius': 0.1}, 'sphere':{'radius': 0.2}})
    p3Dw.zoomTo()
    p3Dw.update()
    select_mo.description = "MO number"

def generate_cubefiles():
    pyscf.tools.cubegen.density(mol, "electron_density.cube", mf.make_rdm1()) # create electron density
    for mo_level in range(len(mf.mo_energy)):
        pyscf.tools.cubegen.orbital(mol, "orb_num_"+str(mo_level)+".cube", mf.mo_coeff[:,mo_level])
    select_mo.max = len(mf.mo_energy)
    
def classical_settings_confirmed(_):
    start_step_3a()
    global mol
    global conv_params
    global mf
    mol.basis = select_basis.value
    mol.spin  = select_spin.value
    mol.symmetry = select_symmetry.value
    mol.charge = select_charge.value
    mol.verbose = {"Minimal": 1, "Optimal": 3, "Full": 5}[select_verbosity.value]
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
    classical_energy.value = "Calculating..."
    with capture_output() as io:
        mol = geomoptimize(mf, **conv_params)
    mol.tofile("molecule.xyz",format="xyz") # final optimized geometry
    mf.kernel() # setup kernel again
    with pyscf_out:
        io.show()
    classical_energy.value = "Calculation done, now generating visulization files."
    p3Dw.removeAllModels()
    p3Dw.removeAllShapes() # just to be sure
    p3Dw.addModel(open("molecule.xyz",'r').read(), "xyz")
    p3Dw.setStyle({'stick':{'radius': 0.2}, 'sphere':{'radius': 0.3}})
    p3Dw.update()
    generate_cubefiles()
    classical_energy.value = "The classically calculated ground state energy of the system is: "+str(mf.e_tot)+" hartree."
    classical_result_label.value = "$Download$ $Result$ $File$:"
    with classical_result_link:
        display(FileLink(mol.output))
    start_step_3b()

# For Step 3




# Observers

file_picker.observe(file_picker_switch, names='value') # Monitor option chosen by file picker
file_upload.observe(upload_button_used, names='value') # Monitor which file was picked
confirm_file_button.on_click(file_confirmed) # To move to step 2
confirm_classical_settings.on_click(classical_settings_confirmed) # To move to step 3
select_visual.observe(visual_switched, names='value') # Monitor charge density or MO toggle
select_mo.observe(mo_changed, names='value') # to set the MO in the visualization

# Build the full widget
calculator1 = widgets.VBox(children=[titlebar,
                                    subtitle1,
                                    file_box,confirm_file_button,
                                    subtitle2,
                                    select_basis, select_spin, select_symmetry,select_charge,
                                    select_method, select_geooptimizer,
                                    widgets.Label(value="$Convergence$ $settings $"),
                                    select_conv_params, select_verbosity, confirm_classical_settings,
                                    subtitle3, classical_energy, download_classical_result])
calculator2 = widgets.VBox(children=[visualization,
                                    errorbar, pyscf_out])


# In[2]:


display(calculator1)
p3Dw.show()
display(calculator2)

