import pytest
import numpy as np
import csdl_alpha as csdl
from VortexAD import VortexLatticeMethod
from VortexAD.utils.meshing.gen_vlm_mesh import gen_vlm_mesh

@pytest.mark.parametrize("test_input", [1, 2, 3])
def test_dummy_function(test_input):
    assert test_input > 0

def test_vlm_mesh():
    ns, nc = 11, 3
    b, c = 10., 1.
    vlm_mesh = gen_vlm_mesh(ns, nc, b, c)
    assert vlm_mesh.shape == (nc, ns, 3)

def test_vlm():
    # instantiate recorder to assemble the graph
    recorder = csdl.Recorder(inline=False)
    recorder.start()

    ns, nc = 11, 3
    b, c = 10., 1.
    mesh_orig = gen_vlm_mesh(ns, nc, b, c)
    mesh = csdl.Variable(value=mesh_orig).expand((1, nc, ns, 3), 'ijk->aijk')
    mesh_list = [mesh]

    pitch = csdl.Variable(value=np.array([5.]))

    input_dict = {
        'V_inf': 10.,
        'alpha': pitch,
        'meshes': mesh_list
    }

    vlm = VortexLatticeMethod(
        input_dict
    )
    vlm_outputs = ['surface_lift', 'surface_CL', 'surface_CDi', 'gamma', 'wake_vortex_mesh', 'net_gamma']
    vlm_outputs.append('surface_panel_forces')
    vlm.declare_outputs(vlm_outputs)

    outputs = vlm.evaluate()
    CL = outputs['surface_CL'][0]
    CDi = outputs['surface_CDi'][0]

    # csdl-jax stuff
    inputs = [pitch]
    outputs = [CL, CDi]

    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=inputs,
        additional_outputs=outputs,
        gpu=False
    )
    sim.run()
    CL_val = sim[CL]
    CDi_val = sim[CDi]

    print(CL_val)
    print(CDi_val)

    CL_val_oas = np.array([0.4426841725811703])
    CDi_val_oas = np.array([0.005878842561184834])
    assert np.isclose(CL_val, CL_val_oas, rtol=1e-03)
    assert np.isclose(CDi_val, CDi_val_oas, rtol=1e-03)