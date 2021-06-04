#!/usr/bin/env python
# coding: utf-8

# # Welcome to Q³
# ## Qiskit QuESt Qalculator
# QuESt, or Quantum Electronic Structure, is the new way of doing chemistry!
# Happy Qomputing!

# To launch the application, please hover on the `rocket` icon (top right) and then click on `Live Code`.  
# 
# Then please wait a while till the backend is ready :)
# The setup time is small for any subsequent launches.  
# ![Instructions](LiveLaunchScreenshot.png "Instructions")
# 
# The notebook also allows you to visualize molecular orbitals on the fly.  
# However, the online backend we have embedded here at the moment is not powerful enough.  
# To enable visualizations, you download the notebook and set `visual = True` to use the functionality.

# In[1]:


# CODE BEGINS

visual = True

import ipywidgets as widgets
import pyscf, py3Dmol
import pathlib, io, time
import numpy as np
import matplotlib.pyplot as plt
from pyscf.geomopt.geometric_solver import optimize as geomoptimize
from pyscf.geomopt.berny_solver import optimize as bernyoptimize
from IPython.display import FileLink
from IPython.utils.io import capture_output
from energydiagram import ED

# Setting some defaults
molecule_file = None
default_selection_width = "60%"
style = {'description_width': 'initial'}
hbegin = "<"+"h2"+">" # using '+' to avoid HTML look in an unrendered ipynb
hend = "<"+"/h2"+">"

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
global mol, mf, es_data
es_data = {}
mol = pyscf.gto.Mole()
mf = None
mol.atom = stored_geometries[default_geometry]
mol.unit = 'A'
mol.output = "geoopt.output"

# Starting with UI

# All widgets logic

titlebar = widgets.HTML(value=hbegin.replace('2','1')+"Qiskit QuESt Calculator"+hend.replace('2','1'))

errorbar = widgets.HTML(value="")
pyscf_out = widgets.Output()

def error_occured(error_message):
    errorbar.value = error_message
    exit()
    time.sleep(10) # Give kernel time to exit
    

# For Step 1: Selecting a molecule

# Program Logic

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

def plot_mo_diagram():
    global es_data
    degeneracy_threshold = 0.2
    levels = [1]
    if es_data["spin"]:
        mo_energy = np.sort((np.vstack(mf.mo_energy)).flatten())[::2]
    else:
        mo_energy = mf.mo_energy
    for diff in np.abs(np.diff(mo_energy)):
        if diff < degeneracy_threshold:
            levels[-1] = levels[-1]+1
            continue
        else:
            levels.append(1)   
    es_data["mo_degeneracy"] = levels[:] # deep copy

    # separate into filled and unfilled
    plot_filled = []
    counted_mo = 0
    while counted_mo < es_data['num_of_occ']:
        _ = levels.pop(0)
        counted_mo += _
        plot_filled.append(_)
    plot_unfilled = levels

    if es_data["elec_above_homo"]:
        electrons_left = es_data["elec_above_homo"]
        unocc_mos_required = 1
        for level in plot_unfilled:
            electrons_left = electrons_left - (level*2)
            if electrons_left <= 0:
                select_num_unocc_mo.min = unocc_mos_required
                break
            else:
                unocc_mos_required += 1

    # setup labels and separation
    plot_energy = list(range(-5,25,5)) #energy from -5 to 20, basically determines separation
    mo_labels = ["HOMO-1", "HOMO", "LUMO","LUMO+1","LUMO+2","LUMO+3"]

    # have to modify rcParams because the newer version of ED is buggy: doesn't return ax and plt for manipulation
    fig_height = len(es_data["mo_degeneracy"])*2
    if fig_height > 12:
        fig_height = 12
    fig_width  = 6
    plt.rcParams['figure.figsize'] = (len(es_data["mo_degeneracy"])*2,12)
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelcolor'] = 'w'
    
    # plot from HOMO-1 to LUMO+3 (or less)
    homo_include = 2
    if es_data["num_of_occ"] == 1 or es_data["num_of_occ"] == 0:
        homo_include = 1
        mo_labels.pop(0)
    mo_diagram = ED()
    for i in range(homo_include): # for HOMO-1 and HOMO
        mo_diagram.add_level(plot_energy[i], mo_labels[i])
        if es_data["num_of_elec"] == 1:
            mo_diagram.add_electronbox(level_id=i, boxes=1, electrons=1, side=1.5, spacing_f=2.5)
        else:
            mo_diagram.add_electronbox(level_id=i, boxes=plot_filled[i-homo_include], electrons=(plot_filled[i-homo_include])*2, side=1.5, spacing_f=2.5)

    for i in range(homo_include, len(plot_unfilled)+homo_include, 1): # for all the LUMOs
        mo_diagram.add_level(plot_energy[i], mo_labels[i])
        if i == homo_include:
            mo_diagram.add_electronbox(level_id=i, boxes=plot_unfilled[i-homo_include], electrons=es_data["elec_above_homo"], side=1.5, spacing_f=2.5)
        else:
            mo_diagram.add_electronbox(level_id=i, boxes=plot_unfilled[i-homo_include], electrons=0, side=1.5, spacing_f=2.5)
        if i == homo_include+1:
            break # break after reaching (and printing) to LUMO+1; can be easily modified to print all LUMOs
    mo_diagram.offset = -0.7
    with capture_output() as mo_diag_ouput:
        mo_diagram.plot(ylabel="")
    with mo_diag_view:
        mo_diag_ouput.show()
    plt.rcParams.update(plt.rcParamsDefault) # restore usual settings

def generate_cubefiles():
    pyscf.tools.cubegen.density(mol, "electron_density.cube", mf.make_rdm1()) # create electron density
    for mo_level in range(es_data["num_of_mo"]):
        pyscf.tools.cubegen.orbital(mol, "orb_num_"+str(mo_level)+".cube", mf.mo_coeff[:,mo_level])
    select_visual_mo.max = es_data["num_of_mo"]

# Widgets

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

# Widget logic

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
    if visual:
        p3Dw.removeAllModels()
        p3Dw.removeAllShapes() # just to be sure
        p3Dw.addModel(open("given_molecule.xyz",'r').read(), "xyz")
        p3Dw.setStyle({'stick':{'radius': 0.2}, 'sphere':{'radius': 0.3}})
        p3Dw.update()
    del temp_mol
    end_step_1()
    
def end_step_1():
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
    select_classical_screen_output.disabled = False
    for widget in get_custom_settings:
        widget.disabled = False
    confirm_classical_settings.disabled = False

# For Step 2 : Doing classical calculation

# Widgets

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
                                        value = True, description="Use Point Group Symmetry",
                                        style=style, layout=widgets.Layout(width=default_selection_width))

select_charge = widgets.SelectionSlider(options=[-1,0,1], disabled=True,
                                        value = 0, description="Charge on the molecule",
                                        style=style, layout=widgets.Layout(width=default_selection_width))

select_method = widgets.Dropdown(options=[('Hartree-Fock (HF)', 'HF'),
                                          ('Restricted closed-shell HF (RHF)','RHF'),
                                          ('Restricted open-shell HF (ROHF)','ROHF')],
                                 disabled=True, value = 'HF', description="Method",
                                 style=style, layout=widgets.Layout(width=default_selection_width))

select_geooptimizer = widgets.Dropdown(options=[('geomeTRIC','geometric'),('PyBerny', 'pyberny')], disabled=True,
                                       value = 'geometric', description="Geometry Optimizer",
                                       style=style, layout=widgets.Layout(width=default_selection_width))

criteria_description = {'convergence_energy':"Energy [hartree]",
                        'convergence_grms': "Gradient (RMS) [hartree/bohr]",
                        'convergence_gmax': "Gradient (maximum)  [hartree/bohr]",
                        'convergence_drms': "Drift (RMS) [angstrom]",
                        'convergence_dmax': "Gradient (maximum) [angstrom]",
                       }

default_settings = []
for criteria in conv_params["default"].keys():
    default_settings.append(widgets.FloatText(value=conv_params["default"][criteria],
                                              description =  criteria_description[criteria],
                                              style=style, disabled=True, layout=widgets.Layout(width="99%")))
default_settings = widgets.VBox(children=default_settings)

tight_settings = []
for criteria in conv_params["tight"].keys():
    tight_settings.append(widgets.FloatText(value=conv_params["tight"][criteria],
                                            description =  criteria_description[criteria],
                                            style=style, disabled=True, layout=widgets.Layout(width="99%")))
tight_settings = widgets.VBox(children=tight_settings)

get_custom_settings = []
for criteria in conv_params["default"].keys():
    get_custom_settings.append(widgets.FloatText(value=conv_params["default"][criteria],
                                                 description =  criteria_description[criteria],
                                                 style=style, disabled=True, layout=widgets.Layout(width="99%")))
custom_input = widgets.VBox(children=get_custom_settings)

select_conv_params = widgets.Tab(children = [default_settings, tight_settings, custom_input], disabled=True,
                                 layout=widgets.Layout(width=default_selection_width))
select_conv_params._titles = {0:"Default criteria",1:"Tight criteria", 2:"Custom criteria"}

select_classical_screen_output = widgets.Checkbox(description="Show on-screen output",
                                                  value=False, disabled=True, indent=True,
                                                  style=style, layout=widgets.Layout(width="39%"))

select_verbosity = widgets.SelectionSlider(description="Output Verbosity", options=["Minimal", "Optimal", "Full"],
                                           value="Optimal", disabled=True,
                                           style=style, layout=widgets.Layout(width="default_selection_width"))
                                           

classical_output = widgets.HBox(children=[select_classical_screen_output, select_verbosity],
                                style=style, layout=widgets.Layout(width=default_selection_width))

confirm_classical_settings = widgets.Button(description="Run Classical Calculation", button_style="success",
                                  disabled=True, layout=widgets.Layout(width=default_selection_width))

# Widget Logic and Program Logic

def classical_settings_confirmed(_):
    end_step_2() # at the beginning as user should not be able to change anything once calculation starts
    global mol
    global conv_params
    global mf
    global es_data
    mol.basis = select_basis.value
    mol.spin  = select_spin.value
    mol.symmetry = select_symmetry.value
    mol.charge = select_charge.value
    mol.verbose = {"Minimal": 1, "Optimal": 3, "Full": 5}[select_verbosity.value]
    try: # to check validity of uploaded geometry / included geometry and to generate xyz for 3D view
        mol.build()
    except:
        error_occured("Error occured when building. Please check input sanity, especially spin configuration!")
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
    elif select_method.value == 'RHF':
        mf = pyscf.scf.RHF(mol)
    elif select_method.value == 'ROHF':
        mf = pyscf.scf.ROHF(mol)
    classical_energy.value = "Calculating..."
    with capture_output() as pyscf_screen_output:
        try:
            mol = geomoptimize(mf, **conv_params)
        except:
            pass # eror will be printed on screen output
    mf.kernel() # setup kernel again
    select_classical_screen_output.disabled = True
    if mf.e_tot == 0:
        # PySCF calculation has failed, therefore force screen output and exit
        select_classical_screen_output.value = True        
    # User has time till this point to show output on screen; can be helpful if PySCF is taking super long
    if select_classical_screen_output.value:
        with pyscf_out:
            pyscf_screen_output.show()
    if mf.e_tot == 0:
        error_occured("PySCF calculation failed! Make sure that your input parameters were valid.")
        return
    mol.tofile("molecule.xyz",format="xyz") # final optimized geometry
    es_data["spin"] = True if type(mf.mo_occ) is tuple else False
    if es_data["spin"]:
        mo_occ = np.sum(np.vstack(mf.mo_occ), axis=0)
    else:
        mo_occ = mf.mo_occ
    es_data["num_of_mo"] = len(mo_occ)
    es_data['num_of_elec'] = sum(mol.nelec)
    if es_data['num_of_elec'] == 1:
        es_data['num_of_occ'] = int(mf.mo_occ[0])
        es_data['num_of_unocc'] = 0
        es_data['num_of_parocc'] = int(mf.mo_occ[1])
        es_data['elec_above_homo'] = int(mf.mo_occ[1])
    else:
        es_data['num_of_occ'] = int(np.sum(np.floor(mo_occ/2)))
        es_data['num_of_unocc'] = es_data["num_of_mo"] - np.count_nonzero(mo_occ)
        es_data['num_of_parocc'] = es_data["num_of_mo"] - es_data['num_of_unocc'] - es_data['num_of_occ']
        es_data['elec_above_homo'] = es_data['num_of_elec']-(es_data['num_of_occ']*2)
    if es_data['num_of_elec'] == 1:
        select_num_unocc_mo.max = 1
    else:
        select_num_unocc_mo.max = es_data['num_of_unocc']+es_data['num_of_parocc']
    if es_data['num_of_occ'] == 0:
        select_num_occ_mo.min = 0
    select_num_occ_mo.max   = es_data['num_of_occ']
    classical_energy.value = "The classically calculated ground state energy of the system is: "+str(mf.e_tot)+" hartree."
    classical_result_label.value = "$Download$ $Result$ $File$:"
    with classical_result_link:
        display(FileLink(mol.output))
    plot_mo_diagram()
    mo_diag_header.value="A partial Molecular Occupation diagram is below. Use it to guide selection of Active Space."
    proceed_to_quantum.disabled = False
    if visual:
        select_visual_mo.options = get_visual_mo_options()
        select_visual_mo.value = es_data["num_of_occ"]
        classical_energy.value = "Calculation done, now generating visulization files."
        p3Dw.removeAllModels()
        p3Dw.removeAllShapes() # just to be sure
        p3Dw.addModel(open("molecule.xyz",'r').read(), "xyz")
        p3Dw.setStyle({'stick':{'radius': 0.2}, 'sphere':{'radius': 0.3}})
        p3Dw.update()
        generate_cubefiles()
        select_visual.disabled = False

def end_step_2():
    select_basis.disabled = True
    select_spin.disabled = True
    select_symmetry.disabled = True
    select_charge.disabled = True
    select_method.disabled = True
    select_geooptimizer.disabled = True
    for widget in get_custom_settings:
        widget.disabled = True
    confirm_classical_settings.unobserve_all()
    confirm_classical_settings.disabled = True
    if select_verbosity.value: # Disable if output was asked for
        select_verbosity.disabled = True

# For Step 3 : Visual analysis, no calculations other than computation of cube files if required

# 3D viewer : Logic and Widgets

subtitle3 = widgets.HTML(value=hbegin+"Analyze Classical Results"+hend)

classical_energy = widgets.Label(value="")

if visual:
    p3Dw = py3Dmol.view(width=600,height=400)

select_visual = widgets.Dropdown(options=["Geometry","Charge Density","Molecular Orbital"], disabled=True,
                                 value = "Geometry", description="Plot:",
                                 style=style, layout=widgets.Layout(width=default_selection_width))

redraw_view = widgets.Button(description="redraw view", disabled = True,
                             style=style, layout=widgets.Layout(width=default_selection_width))

download_view3D = widgets.Button(description="download view", disabled = True,
                                     style=style, layout=widgets.Layout(width=default_selection_width))

def get_visual_mo_options():
    homos = ["HOMO"]+["HOMO-"+str(_) for _ in range(1, 101, 1)]
    homos.reverse()
    options = homos[-es_data["num_of_occ"]:]+["LUMO"]
    for i in range(es_data["num_of_mo"]-len(options)):
        options.append("LUMO+"+str(i+1))
    return list(zip(options,range(1,(len(options)+1),1)))

select_visual_mo = widgets.SelectionSlider(description="Select MO: ", options=[("HOMO", 1)], value=1, disabled=True, 
                                     style=style, layout=widgets.Layout(width=default_selection_width))

iso_properties = {'isoval':  0.03, 'color':"yellow", 'opacity':0.5}
iso_properties_neg_color = "green"

set_isovalue = widgets.BoundedFloatText(value=0.03, min = 0, max = 999, step=0.01, description="isovalue", disabled=True,
                             style=style, layout=widgets.Layout(width="30%"))

set_opacity  = widgets.BoundedFloatText(value=0.5, min=0, max=1.0, step=0.05, description="opacity", disabled=True,
                             style=style, layout=widgets.Layout(width="30%"))

select_color_pos = widgets.ColorPicker(description=" + ", concise=True, value='yellow', disabled=True,
                                       style=style, layout=widgets.Layout(width="10%"))
select_color_neg = widgets.ColorPicker(description=" − ", concise=True, value='green', disabled=True,
                                       style=style, layout=widgets.Layout(width="10%"))

visualization = widgets.VBox(children=[widgets.HBox(children=[select_visual, download_view3D, redraw_view],
                                                    style=style, layout=widgets.Layout(width=default_selection_width)),
                                       widgets.VBox(children=[select_visual_mo]),
                                       widgets.HBox(children=[set_isovalue, select_color_pos, select_color_neg, set_opacity],
                                                    style=style, layout=widgets.Layout(width=default_selection_width))])

def freeze_surface3d(value=True):
    set_isovalue.disabled = value
    select_color_pos.disabled = value
    select_color_neg.disabled = value
    set_opacity.disabled = value
    redraw_view.disabled = value

def visual_switched(value):
    download_view3D.disabled = True
    select_visual_mo.disabled = True
    freeze_surface3d(True)
    if value is None: # to make it work with visual_switched_coz_redraw()
        value = {"new": select_visual.value}
    if value['new'] == "Geometry":
        p3Dw.removeAllShapes() # just to be sure
        p3Dw.setStyle({'stick':{'radius': 0.2}, 'sphere':{'radius': 0.3}})
        p3Dw.zoomTo()
        p3Dw.update()
    elif value['new'] == "Charge Density":
        redraw_view.description = "loading..."
        cube_data = open("electron_density.cube").read()
        p3Dw.removeAllShapes()
        p3Dw.addVolumetricData(cube_data, "cube", iso_properties)
        p3Dw.setStyle({'stick':{'radius': 0.1}, 'sphere':{'radius': 0.2}})
        p3Dw.zoomTo()
        p3Dw.update()
        redraw_view.description = "redraw view"
        freeze_surface3d(False)
        select_color_neg.disabled = True
    elif value['new'] == "Molecular Orbital":
        mo_changed(None)
    download_view3D.disabled = False

def visual_switched_coz_redraw(_):
    visual_switched(None)
    if select_visual.value == "Geometry":
        pass
    elif select_visual.value == "Charge Density":
        set_isovalue.value = iso_properties['isoval']
        set_opacity.value = iso_properties['opacity']
        select_color_pos.value = iso_properties['color']
    elif select_visual.value == "Molecular Orbital":
        set_isovalue.value = iso_properties['isoval']
        set_opacity.value = iso_properties['opacity']
        select_color_pos.value = iso_properties['color']
        select_color_neg.value = iso_properties_neg_color
        
def mo_changed(_):
    select_visual_mo.disabled = True
    redraw_view.description = "loading..."
    cube_data = open("orb_num_"+str(select_visual_mo.value-1)+".cube").read()
    p3Dw.removeAllShapes()
    p3Dw.addVolumetricData(cube_data, "cube", iso_properties)
    p3Dw.addVolumetricData(cube_data, "cube",
                           {"isoval": -iso_properties['isoval'],
                            "opacity": iso_properties['opacity'],
                            "color": iso_properties_neg_color})
    p3Dw.setStyle({'stick':{'radius': 0.1}, 'sphere':{'radius': 0.2}})
    p3Dw.zoomTo()
    p3Dw.update()
    redraw_view.description = "redraw view"
    select_visual_mo.disabled = False
    freeze_surface3d(False)

def isovalue_setter(_):
    iso_properties['isoval'] = set_isovalue.value
        
def opacity_setter(_):
    iso_properties['opacity'] = set_opacity.value

def pos_color_setter(_):
    iso_properties['color'] = select_color_pos.value
    
def neg_color_setter(_):
    iso_properties_neg_color = select_color_neg.value

def download_view(_):
    p3Dw.png()

# Widgets (for Step 3) (other than 3D viewer)

classical_result_label = widgets.Label(value="")
classical_result_link = widgets.Output()

download_classical_result = widgets.HBox(children=[classical_result_label, classical_result_link],
                                        layout=widgets.Layout(width=default_selection_width))

mo_diag_header = widgets.Label(value="")
mo_diag_view = widgets.Output()

# Quantum Part

# imports for Quantum calculator

from qiskit_nature.drivers import PySCFDriver, UnitsType, HFMethodType
from qiskit_nature.drivers import Molecule as QMolecule
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.transformers import ActiveSpaceTransformer
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.mappers.second_quantization import BravyiKitaevMapper, JordanWignerMapper
from qiskit_nature.circuit.library import HartreeFock, UCCSD
from qiskit.circuit.library import EfficientSU2
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import NumPyMinimumEigensolverFactory
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.algorithms import optimizers, VQE
from qiskit import Aer

# widgets

subtitle4 = widgets.HTML(value=hbegin+"Setup Quantum Calculation"+hend)

quantum_status = widgets.Label(value="")

proceed_to_quantum  = widgets.Button(description="Start Quantum Calculation", button_style="success", disabled=True,
                                     style=style, layout=widgets.Layout(width=default_selection_width))

select_num_unocc_mo = widgets.IntSlider(description="# of Unoccupied MO for AS",   value=1, min=1, max=1, disabled=True,
                                        style=style, layout=widgets.Layout(width=default_selection_width))

select_num_occ_mo   = widgets.IntSlider(description="# of Occupied MO for AS", value=1, min=1, max=1, disabled=True,
                                        style=style, layout=widgets.Layout(width=default_selection_width))

describe_AS         = widgets.Label("")

select_mapper       = widgets.Dropdown(options=["Jordan-Wigner","Bravyi-Kitaev"], disabled=True,
                                       value="Jordan-Wigner", description = "Parity Mapper",
                                       style=style, layout=widgets.Layout(width=default_selection_width))

select_two_qubit_reduction = widgets.ToggleButtons(options=[True,False], disabled=True,
                                                   value = True, description="Use Two Qubit Reduction",
                                                   style=style, layout=widgets.Layout(width=default_selection_width))

select_ansatz       = widgets.Dropdown(options=[("Physically Motivated (UCCSD)",'UCCSD'),
                                                ("Heuristic (Efficient SU2)",'ESU2')], disabled=True,
                                       value="UCCSD", description = "Ansatz",
                                       style=style, layout=widgets.Layout(width=default_selection_width))

select_backend      = widgets.Dropdown(options=[("IBM Simulator",'sv_simul')], disabled=True,
                                       value="sv_simul", description = "Select your Quantum Computer",
                                       style=style, layout=widgets.Layout(width=default_selection_width))

select_optimizer    = widgets.Dropdown(options=['COBYLA', 'L_BFGS_B', 'SLSQP', 'SPSA'], disabled=True,
                                       value="L_BFGS_B", description = "Optimizer",
                                       style=style, layout=widgets.Layout(width=default_selection_width))

select_solver       = widgets.Dropdown(options=[("Classical solver",'numpy'),
                                                ("VQE",'vqe'),
                                                ("Classical and VQE",'both')], disabled=True,
                                       value="both", description="Solver",
                                       style=style, layout=widgets.Layout(width=default_selection_width))
confirm_quantum_settings = widgets.Button(description="Confirm Quantum Calculation Options", disabled=True,
                                          button_style="success",
                                          style=style, layout=widgets.Layout(width=default_selection_width))

quantum_output      = widgets.Output()


def deactivate_quantum_options(value=True):
    select_num_unocc_mo.disabled = value
    select_num_occ_mo.disabled = value
    select_mapper.disabled = value
    select_two_qubit_reduction.disabled = value
    select_ansatz.disabled = value
    select_backend.disabled = value
    select_optimizer.disabled = value
    select_solver.disabled = value
    confirm_quantum_settings.disabled = value
    if not value: # i.e. the setup is activated
        changed_AS(None)
    
def changed_AS(_):
    nelec = select_num_occ_mo.value*2+es_data["num_of_parocc"]
    nocc  = select_num_occ_mo.value
    nunocc  = select_num_unocc_mo.value
    describe_AS.value = "Current AS selection has "+str(nelec)+" electrons and would require as many qubits."

# Setup Calculation

global qmol, driver, exact_result, vqe_result
exact_result = None
vqe_result = None
# some quantum variables are global otherwise there is a lot of passing around

def setup_quantum(_):
    proceed_to_quantum.disabled = True
    proceed_to_quantum.unobserve_all()
    quantum_status.value="Building quantum molecule"
    global qmol, driver
    picked_HF = {'HF':HFMethodType.UHF,
                 'RHF':HFMethodType.RHF,
                 'ROHF':HFMethodType.ROHF}[select_method.value]
    qmol          = QMolecule(geometry=mol.atom, charge=mol.charge, multiplicity=mol.multiplicity)
    driver        = PySCFDriver(molecule=qmol, unit=UnitsType.BOHR, basis=mol.basis, hf_method=HFMethodType.RHF)
    qmol      = driver.run()
    quantum_status.value="Qmolecule built. Select calculation options"
    deactivate_quantum_options(False)
    
def quantum_settings_confirmed(_):
    global qmol, driver, exact_result, vqe_result
    deactivate_quantum_options(True)
    confirm_quantum_settings.unobserve_all()
    select_num_occ_mo.unobserve_all()
    select_num_unocc_mo.unobserve_all()
    quantum_status.value = "Starting calculation"
    num_electrons_AS = (select_num_occ_mo.value*2)
    if es_data['elec_above_homo']: # all above homo have to be included
        num_electrons_AS += es_data['elec_above_homo']
    if (num_electrons_AS % 2 != 0):
        tup_len =  num_electrons_AS // 2
        num_electrons_AS = ((tup_len+1),(tup_len))
    active_space  = ActiveSpaceTransformer(num_electrons=num_electrons_AS,
                                           num_molecular_orbitals=(select_num_occ_mo.value+select_num_unocc_mo.value))
    es_problem    = ElectronicStructureProblem(driver, q_molecule_transformers=[active_space])
    second_q_ops  = es_problem.second_q_ops()
    main_op       = second_q_ops[0] # just the hamiltonian (energy)
    if select_mapper.value == "Jordan-Wigner":
        mapper    = JordanWignerMapper()
    elif select_mapper.value == "Bravyi-Kitaev":
        mapper    = BravyiKitaevMapper()
    converter     = QubitConverter(mapper=mapper, two_qubit_reduction=select_two_qubit_reduction.value)
    num_particles = (es_problem.molecule_data_transformed.num_alpha, es_problem.molecule_data_transformed.num_beta)
    qubit_op      = converter.convert(main_op, num_particles=num_particles)
    num_spin_orbitals = 2 * es_problem.molecule_data_transformed.num_molecular_orbitals
    if select_ansatz.value == "UCCSD":
        init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
        ansatz     = UCCSD(converter, num_particles, num_spin_orbitals, initial_state=init_state)
    elif select_ansatz.value == "ESU2":
        ansatz = EfficientSU2(num_qubits=qubit_op.num_qubits, reps=3, entanglement='full', insert_barriers=True)
    with capture_output() as qinit:
        if select_ansatz.value == "UCCSD":        
            print("\n\nInitial State                     \n=============")
            print(init_state)
        print("\n\nAnsatz                 \n======")
        print(ansatz)
    if select_backend.value == "sv_simul":
        backend = Aer.get_backend('statevector_simulator')
    # You may want to tune the parameters of each optimizer if you download the notebook,
    # for the GUI these the defaults are used:
    if select_optimizer.value == 'COBYLA':
        optimizer = optimizers.COBYLA(maxiter=500)
    elif select_optimizer.value == 'L_BFGS_B':
        optimizer = optimizers.L_BFGS_B(maxfun=500)
    elif select_optimizer.value == 'SPSA':
        optimizer = optimizers.SPSA(maxiter=500)
    elif select_optimizer.value == 'SLSQP':
        optimizer = optimizers.SLSQP(maxiter=500)
    
    if select_solver.value == "numpy" or select_solver.value == "both":
        quantum_status.value = "Solving using classical solver"
        solver = NumPyMinimumEigensolverFactory()
        calc = GroundStateEigensolver(converter, solver)
        exact_result = calc.solve(es_problem)
        exact_energy = np.real(exact_result.eigenenergies[0])
        with capture_output() as qout_result_1:
            print("\n\nResult from the classical/exact solver follow:                     \n==============================================")
            print(exact_result)
    else:
        with capture_output() as qout_result_1:
             print("\n\nClassical/exact solver not used.")
            

    if select_solver.value == "vqe" or select_solver.value == "both":
        try:
            initial_point = [0.01] * len(ansatz.ordered_parameters)
        except:
            initial_point = [0.01] * ansatz.num_parameters
        quantum_status.value = "Solving using VQE solver"
        algorithm = VQE(ansatz,
                        optimizer=optimizer,
                        quantum_instance=backend,
                        initial_point=initial_point)
        vqe_result = algorithm.compute_minimum_eigenvalue(qubit_op)
        with capture_output() as qout_result_2:
            print("\n\nResult from the VQE Solver follow:                     \n=================================")
            print(vqe_result)
        vqe_energy = vqe_result.eigenvalue
    else:
        with capture_output() as qout_result_2:
            print("\n\nVQE solver not used.")
    with quantum_output:
        print("Energy component from Active State is:")
        if exact_result:
            print("Exact solver: ",exact_result.computed_energies[0])
        if vqe_result:
            print("VQE solver:   ",vqe_result.eigenvalue)
        if exact_result and vqe_result and select_backend.value=="sv_simul":
            print("Don't be surprised that they are similar: we have used a simulator :)")
        qout_result_1.show()
        qout_result_2.show()
        qinit.show()
        
# Observers

file_picker.observe(file_picker_switch, names='value') # Monitor option chosen by file picker
file_upload.observe(upload_button_used, names='value') # Monitor which file was picked
confirm_file_button.on_click(file_confirmed) # To move to step 2
confirm_classical_settings.on_click(classical_settings_confirmed) # To move to step 3
# ^^^ Unobserved at end of Step 1

select_visual.observe(visual_switched, names='value')
redraw_view.on_click(visual_switched_coz_redraw) # Monitor charge density or MO toggle
select_visual_mo.observe(mo_changed, names='value') # to set the MO in the visualization
set_isovalue.observe(isovalue_setter)
set_opacity.observe(opacity_setter)
select_color_pos.observe(pos_color_setter)
select_color_neg.observe(neg_color_setter)
download_view3D.on_click(download_view)
# ^^^ We never unoberve these as they constitute output

proceed_to_quantum.on_click(setup_quantum)
select_num_unocc_mo.observe(changed_AS)
select_num_occ_mo.observe(changed_AS)
confirm_quantum_settings.on_click(quantum_settings_confirmed)


# Build the full widget
calculator1 = widgets.VBox(children=[titlebar,
                                    subtitle1,
                                    file_box,confirm_file_button,
                                    subtitle2,
                                    select_basis, select_spin, select_charge, select_symmetry,
                                    select_method, select_geooptimizer,
                                    widgets.Label(value="$Convergence$ $settings $"),
                                    select_conv_params, classical_output, confirm_classical_settings,
                                    subtitle3, classical_energy, download_classical_result,
                                    mo_diag_header, mo_diag_view])


calculator2_children  = [visualization] if visual else []
calculator2_children += [subtitle4, quantum_status, proceed_to_quantum,
                         select_num_unocc_mo, select_num_occ_mo, describe_AS,
                         select_mapper, select_two_qubit_reduction, select_ansatz,
                         select_backend, select_optimizer, select_solver, confirm_quantum_settings,
                         errorbar, pyscf_out, quantum_output]

calculator2 = widgets.VBox(children=calculator2_children)

def run():
    display(calculator1)
    if visual:
        p3Dw.show()
    display(calculator2)


# In[2]:


run()
# These global variables are available after full execution of the cells and may be used for further exploration
# in the downloaded version of this notebook.
#  - mol, mf (classical PySCF molecule and it's 'mean field' object)
#  - qmol (classical PySCF molecule used for quantum calculation)
#  - exact_result (result from the exact solver, if used)
#  - vqe_result (result from the vqe solver, if used)

