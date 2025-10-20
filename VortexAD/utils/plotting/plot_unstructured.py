import vedo
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show
import numpy as np
from vedo import *
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

def plot_pressure_distribution(mesh, Cp, connectivity, panel_center=None, bounds=None, on='cells', surface_color='white', cmap='jet', interactive=False, top_view=False, front_top_view=False, camera=False, screenshot=False):
    vedo.settings.default_backend = 'vtk'
    axs = Axes(
        xrange=(0,3),
        yrange=(-7.5, 7.5),
        zrange=(0, 5),
    )
    vp = Plotter(
        bg='white',
        # bg2='white',
        # axes=0,
        #  pos=(0, 0),
        offscreen=False,
        interactive=1,
        size=(2500,2500))

    # Any rendering loop goes here, e.g.
    draw_scalarbar = True
    # color = wake_color
    mesh_points = mesh # does not vary with time here

    # vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=1.).linecolor('black')
    vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=1.)
    Cp_color = np.reshape(Cp, (-1,1))
    if bounds:
        Cp_min, Cp_max = bounds[0], bounds[1]
    else:
        Cp_min, Cp_max = np.min(Cp), np.max(Cp)
    # # Cp_min, Cp_max = -0.4, 1.
    # Cp_min, Cp_max = -1.5, 1.
    # vps.cmap(cmap, Cp_color, on='cells', vmin=Cp_min, vmax=Cp_max)
    vps.cmap(cmap, Cp_color, on=on, vmin=Cp_min, vmax=Cp_max)
    vps.add_scalarbar()
    vp += vps
    # vp += __doc__
    # nl = NormalLines(vps, on=on, scale=0.1)
    # vp += nl
    # if panel_center is not None:
    #     arrows = Arrows(panel_center, Cp+panel_center, c='red')
    # else:
    #     arrows = Arrows(mesh, Cp+mesh, c='red')
    # vp += arrows
    
    # wake_points = wake_mesh[:,i,:(i+1),:]
    # # mu_w = np.reshape(sim['system_model.wig.wig.wig.operation.prob.' + 'op_' + surface_name+'_mu_w'][i, 0:i, :], (-1,1))
    # # if absolute:
    # #     mu_w = np.absolute(mu_w)
    # # wake_points = np.concatenate((np.reshape(mesh_points[-1,:,:],(1,ny,3)), wake_points))
    # nx = wake_points.shape[1]
    # ny = wake_points.shape[2]
    # connectivity = []
    # for k in range(nx-1):
    #     for j in range(ny-1):
    #         connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
    # vps = Mesh([np.reshape(wake_points, (-1, 3)), connectivity], c=color, alpha=1)
    # mu_wake_color = np.reshape(mu_wake[:,i,:(i)*(ny-1)], (-1,1))
    # vps.cmap(cmap, mu_wake_color, on='cells', vmin=min_mu, vmax=max_mu)
    # if draw_scalarbar:
    #     vps.add_scalarbar()
    #     draw_scalarbar = False
    # vps.linewidth(1)
    # vp += vps
    # vp += __doc__
    # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
    # video.action(cameras=[cam1, cam1])
    # vp.show(axs, elevation=-60, azimuth=45, roll=-45,
    #         axes=False, interactive=False)  # render the scene
    # vp.show(axs, elevation=-60, azimuth=-90, roll=90,
    #         axes=False, interactive=False, zoom=True)  # render the scene
    if camera:
        vp.show(vps, camera=camera, interactive=interactive, screenshot=screenshot)  # render the scene
    elif top_view:
        vp.show(axs, elevation=0, azimuth=0, roll=90,
                axes=False, interactive=interactive)  # render the scene
    elif front_top_view:
        vp.show(axs, elevation=0, azimuth=-45, roll=90,
                axes=False, interactive=interactive)  # render the scene
    else:
        show(vps, elevation=-45, azimuth=-45, roll=45,
                axes=False, interactive=interactive)  # render the scene
    # video.add_frame()  # add individual frame
    # # time.sleep(0.1)
    # # vp.interactive().close()
    # vp.close_window()

def plot_wireframe(mesh, wake_mesh, surface_data, wake_data, connectivity, wake_connectivity, wake_form, TE_indices,
                   interactive=False, surface_color='gray', cmap='jet', side_view=False, name='sample_gif', backend='imageio'):
    vedo.settings.default_backend = 'vtk'
    nt = surface_data.shape[0]
    axs = Axes(
        xrange=(0,3),
        yrange=(-7.5, 7.5),
        zrange=(0, 5),
    )
    video = Video(name+".mp4", fps=5, backend=backend)
    # first get min and max mu value:
    min_mu_b = np.min(surface_data)
    max_mu_b = np.max(surface_data)
    min_mu_w = np.min(wake_data)
    max_mu_w = np.max(wake_data)

    min_mu = np.min((min_mu_b, min_mu_w))
    max_mu = np.max((max_mu_b, max_mu_w))

    ns = np.max(wake_connectivity[0,:,0::3])+1
    nTp = wake_connectivity.shape[1]

    for i in range(nt):
        vp = Plotter(
            bg='white',
            # bg2='white',
            # axes=0,
            #  pos=(0, 0),
            offscreen=False,
            interactive=1,
            size=(2500,2500))

        # Any rendering loop goes here, e.g.
        draw_scalarbar = True
        mesh_points = mesh[i,:] # does not vary with time here
        vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=1.)
        # vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=1.).linecolor('black')
        mu_color = np.reshape(surface_data[i,:], (-1,1))
        
        # vps.cmap(cmap, mu_color, on='cells', vmin=min_mu, vmax=max_mu)
        vps.cmap(cmap, mu_color, on='cells')
        vps.add_scalarbar()
        vp += vps
        vp += __doc__

        if i > 0:
            wake_points_iter = wake_mesh[i,:,:].reshape((nt, ns, 3))[:i+1,:].reshape((ns*(i+1), 3))
            if wake_form == 'grid':
                wake_conn_iter = wake_connectivity[:i,:,:].reshape((i*nTp, 4))
                vps = Mesh([np.reshape(wake_points_iter, (-1, 3)), wake_conn_iter], c='gray', alpha=1).linecolor('black')
                mu_wake_color = np.reshape(wake_data[i,:(i)*(nTp)], (-1,1))
                vps.cmap(cmap, mu_wake_color, on='cells', vmin=min_mu, vmax=max_mu)
            # if draw_scalarbar:
            #     vps.add_scalarbar()
            #     draw_scalarbar = False
            elif wake_form == 'lines':
                wake_conn_iter = wake_connectivity[:i,:,:]
                wake_conn_lines = np.concatenate((wake_conn_iter[:,:-1,:2], wake_conn_iter[:,-1:,2:][:,:,::-1]), axis=1)



                wake_conn_lines = np.concatenate((wake_conn_iter[:,:,:2], wake_conn_iter[:,:,2:][:,:,::-1]), axis=1)
                # turn this into a list of tuples and then a set 
                line_pts = []
                ns = len(TE_indices)
                for j in range(i):
                    # line_pts.extend([[ind*j, ind*(j+1)] for ind in TE_indices])
                    line_pts.extend([[wake_mesh[i,ind+j*ns,:], wake_mesh[i, ind+(j+1)*ns,:]] for ind in TE_indices])
                vps = Lines(line_pts, c='black')

            vp += vps
            vp += __doc__
        # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
        # video.action(cameras=[cam1, cam1])
        # vp.show(axs, elevation=-60, azimuth=45, roll=-45,
        #         axes=False, interactive=False)  # render the scene
        # vp.show(axs, elevation=-60, azimuth=-90, roll=90,
        #         axes=False, interactive=False, zoom=True)  # render the scene
        1

        if side_view:
            vp.show(axs, elevation=-90, azimuth=0, roll=0,
                    axes=False, interactive=interactive)  # render the scene
        else:
            vp.show(axs, elevation=-45, azimuth=-45, roll=45,
                    axes=False, interactive=interactive)  # render the scene
        video.add_frame()  # add individual frame

    video.close()  # merge all the recorded frames

def plot_wireframe_old(mesh, wake_mesh, mu, mu_wake, connectivity, nt, interactive = False, plot_mirror = True, wake_color='cyan', rotor_wake_color='red', surface_color='gray', cmap='jet', absolute=True, side_view=False, name='sample_gif', backend='imageio'):
    vedo.settings.default_backend = 'vtk'
    axs = Axes(
        xrange=(0,3),
        yrange=(-7.5, 7.5),
        zrange=(0, 5),
    )
    video = Video(name+".mp4", fps=5, backend=backend)
    # first get min and max mu value:
    min_mu_b = np.min(mu)
    max_mu_b = np.max(mu)
    min_mu_w = np.min(mu_wake)
    max_mu_w = np.max(mu_wake)

    min_mu = np.min((min_mu_b, min_mu_w))
    max_mu = np.max((max_mu_b, max_mu_w))

    for i in range(nt-1):

        # min_mu_b = np.min(mu[:,i,:])
        # max_mu_b = np.max(mu[:,i,:])
        # min_mu_w = np.min(mu_wake[:,i,:])
        # max_mu_w = np.max(mu_wake[:,i,:])

        # min_mu = np.min((min_mu_b, min_mu_w))
        # max_mu = np.max((max_mu_b, max_mu_w))
        vp = Plotter(
            bg='white',
            # bg2='white',
            # axes=0,
            #  pos=(0, 0),
            offscreen=False,
            interactive=1,
            size=(2500,2500))

        # Any rendering loop goes here, e.g.
        draw_scalarbar = True
        color = wake_color
        mesh_points = mesh[:, i, :] # does not vary with time here
        nx = mesh_points.shape[1]
        ny = mesh_points.shape[2]
        # connectivity = []
        # for k in range(nx-1):
        #     for j in range(ny-1):
        #         connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
        vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=1.)
        # vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=1.).linecolor('black')
        mu_color = np.reshape(mu[:,i,:], (-1,1))
        
        vps.cmap(cmap, mu_color, on='cells', vmin=min_mu, vmax=max_mu)
        vp += vps
        vp += __doc__
        wake_points = wake_mesh[:,i,:(i+1),:]
        # mu_w = np.reshape(sim['system_model.wig.wig.wig.operation.prob.' + 'op_' + surface_name+'_mu_w'][i, 0:i, :], (-1,1))
        # if absolute:
        #     mu_w = np.absolute(mu_w)
        # wake_points = np.concatenate((np.reshape(mesh_points[-1,:,:],(1,ny,3)), wake_points))
        nx = wake_points.shape[1]
        ny = wake_points.shape[2]
        connectivity_wake = []
        for k in range(nx-1):
            for j in range(ny-1):
                connectivity_wake.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
        vps = Mesh([np.reshape(wake_points, (-1, 3)), connectivity_wake], c=color, alpha=1)
        if i > 0:
            mu_wake_color = np.reshape(mu_wake[:,i,:(i)*(ny-1)], (-1,1))
            vps.cmap(cmap, mu_wake_color, on='cells', vmin=min_mu, vmax=max_mu)
        if draw_scalarbar:
            vps.add_scalarbar()
            draw_scalarbar = False
        vps.linewidth(1)
        vp += vps
        vp += __doc__
        # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
        # video.action(cameras=[cam1, cam1])
        # vp.show(axs, elevation=-60, azimuth=45, roll=-45,
        #         axes=False, interactive=False)  # render the scene
        # vp.show(axs, elevation=-60, azimuth=-90, roll=90,
        #         axes=False, interactive=False, zoom=True)  # render the scene
        if side_view:
            vp.show(axs, elevation=-90, azimuth=0, roll=0,
                    axes=False, interactive=interactive)  # render the scene
        else:
            vp.show(axs, elevation=-45, azimuth=-45, roll=45,
                    axes=False, interactive=interactive)  # render the scene
        video.add_frame()  # add individual frame
        # time.sleep(0.1)
        # vp.interactive().close()
    #     vp.close_window()
    # vp.close_window()
    video.close()  # merge all the recorded frames