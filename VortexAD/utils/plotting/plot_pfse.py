import vedo
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show
import numpy as np
from vedo import *
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

def plot_wireframe(pm_mesh, pm_connectivity, pm_conn_params, vlm_meshes, vlm_connectivities, wake_mesh, wake_connectivities, surface_data, wake_data, 
                   bounds=False, wake_form='grid', interactive=False, camera=False, surface_color='gray', cmap='jet', side_view=False, name='sample_gif', backend='imageio', fps=5):
    vedo.settings.default_backend = 'vtk'
    nt = surface_data.shape[0]
    num_vlm_meshes = len(vlm_meshes)
    vlm_wake_connectivities = wake_connectivities[1:]

    TE_indices = pm_conn_params[0]
    TE_indices_zeroed = pm_conn_params[1]
    wake_edges2cells = pm_conn_params[2]
    pm_wake_connectivity = wake_connectivities[0]

    axs = Axes(
        xrange=(0,3),
        yrange=(-7.5, 7.5),
        zrange=(0, 5),
    )
    video = Video(name+".mp4", fps=fps, backend=backend)
    # first get min and max mu value:
    min_mu_b = np.min(surface_data)
    max_mu_b = np.max(surface_data)
    min_mu_w = np.min(wake_data)
    max_mu_w = np.max(wake_data)

    min_mu = np.min((min_mu_b, min_mu_w))
    max_mu = np.max((max_mu_b, max_mu_w))

    if bounds:
        min_mu = bounds[0]
        max_mu = bounds[1]

    for i in range(nt):
        print('====')
        print(f'making frame {i} of {nt}')
        vp = Plotter(
            bg='white',
            # bg2='white',
            # axes=0,
            #  pos=(0, 0),
            offscreen=False,
            interactive=1,
            size=(2500,2500))
        
        draw_scalarbar = True
        mesh_points = pm_mesh[i,:] # does not vary with time here
        vps = Mesh([np.reshape(mesh_points, (-1, 3)), pm_connectivity], c=surface_color, alpha=1.)
        # vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=1.).linecolor('black')
        num_panels_pm = len(pm_connectivity)
        mu_pm = surface_data[i,:num_panels_pm]
        mu_color = np.reshape(mu_pm, (-1,1))
        
        vps.cmap(cmap, mu_color, on='cells', vmin=min_mu, vmax=max_mu)
        # vps.cmap(cmap, mu_color, on='cells')
        vps.add_scalarbar()
        vp += vps
        vp += __doc__

        bps, bpe = num_panels_pm, num_panels_pm # b panel start/end
        wns, wne = 0, 0 # wake node start/end
        wps, wpe = 0, 0 # wake panel start/end

        for m in range(num_vlm_meshes):
            print(f'mesh {m}')

            mesh_points = vlm_meshes[m][i,:]
            nc, ns = mesh_points.shape[0], mesh_points.shape[1]
            num_body_panels = (nc-1)*(ns-1)

            bpe += num_body_panels

            reshaped_mesh_points = np.reshape(mesh_points, (-1, 3))
            vps = Mesh([reshaped_mesh_points, vlm_connectivities[m].reshape((-1,4))], c=surface_color, alpha=1.).linecolor('black')
            vlm_surf_data = surface_data[i,bps:bpe]
            surf_color = np.reshape(vlm_surf_data, (-1,1))
        
            vps.cmap(cmap, surf_color, on='cells', vmin=min_mu, vmax=max_mu)
            vps.add_scalarbar()
            vp += vps
            vp += __doc__

            bps += num_body_panels

        # plotting wakes
        if i > 0:
            # panel method wake
            nTp = pm_wake_connectivity.shape[1]
            ns_pm = len(TE_indices)
            num_pm_wake_pts = nt*ns_pm
            wake_points_iter = wake_mesh[i,:num_pm_wake_pts,:].reshape((nt, ns_pm, 3))[-(i+1):,:] # NEW METHOD, reorders wake points for actuating geometries
            wake_points_iter[0,:] = pm_mesh[i,TE_indices]
            wake_points_iter = wake_points_iter.reshape((ns_pm*(i+1), 3))

            ns_panels_pm = pm_wake_connectivity.shape[1]
            num_wake_panels_pm = ns_panels_pm*(nt-1)
            pm_wake_data = wake_data[:,:num_wake_panels_pm]
            if wake_form == 'grid':
                wake_conn_iter = pm_wake_connectivity[:i,:,:].reshape((i*nTp, 4)) # OLD METHOD (don't need to replace)
                vps = Mesh([np.reshape(wake_points_iter, (-1, 3)), wake_conn_iter], c='gray', alpha=1).linecolor('black')

                mu_wake_color = np.reshape(pm_wake_data[i,-(i)*(nTp):], (-1,1)) # NEW METHOD
                vps.cmap(cmap, mu_wake_color, on='cells', vmin=min_mu, vmax=max_mu)

            elif wake_form == 'lines':
                ## NEW METHOD
                wpig = wake_points_iter.reshape((i+1, ns_pm, 3))

                line_pts = []
                line_colors = []
                line_edges = []
                for j in range(i):
                    line_pts.extend([[wpig[j,ind,:], wpig[j+1,ind,:]] for ind in TE_indices_zeroed])

                    # line_edges.extend([(ind+(j)*ns, ind+(j+1)*ns) for ind in TE_indices_zeroed]) # OLD METHOD
                    line_edges.extend([(ind+(nt-i-1+j)*ns_pm, ind+(nt-i+j)*ns_pm) for ind in TE_indices_zeroed]) # NEW METHOD
                    # the new method gets the wake elements closest to the TE to furthest away (the role of +j)
                    # starting from some number of timesteps back from the furthest wake element (the role of -i)
                edge_adj_cells = []
                for edge in line_edges:
                    if edge in wake_edges2cells.keys():
                        adj_cells = wake_edges2cells[edge]
                    elif edge[::-1] in wake_edges2cells.keys():
                        adj_cells = wake_edges2cells[edge[::-1]]
                    edge_adj_cells.append(adj_cells)
                    # edge_color = np.average(wake_data[i,])
                line_colors = [np.average(pm_wake_data[i,ind]) for ind in edge_adj_cells]

                vps = Lines(line_pts, lw=3, c='black')
                vps.cmap(cmap, line_colors, on='cells', vmin=min_mu, vmax=max_mu)
            vp += vps
            vp += __doc__

            wns, wne = num_pm_wake_pts, num_pm_wake_pts # wake node start/end
            wps, wpe = num_wake_panels_pm, num_wake_panels_pm # wake panel start/end



            # VLM wake
            for m in range(num_vlm_meshes):
                mesh_points = vlm_meshes[m][i,:]
                nc, ns = mesh_points.shape[0], mesh_points.shape[1]
                num_body_panels = (nc-1)*(ns-1)

                num_surf_wake_nodes = nt*ns
                num_surf_wake_panels = (nt-1)*(ns-1)
                wne += num_surf_wake_nodes
                wpe += num_surf_wake_panels
                wake_mesh_surf = wake_mesh[i,wns:wne].copy().reshape((nt, ns, 3))
                wake_data_surf = wake_data[i,wps:wpe]
                
                # wake_points_iter = wake_mesh_surf[:i+1,:] # OLD METHOD
                wake_points_iter = wake_mesh_surf[-(i+1):,:] # NEW METHOD
                # wake_points_iter[0,:] = mesh_points[TE_indices]
                wake_points_iter[0,:] = mesh_points[-1,:]
                wake_points_iter = wake_points_iter.reshape((ns*(i+1), 3))

                vlm_wake_connectivity = vlm_wake_connectivities[m]
                nTp = vlm_wake_connectivity.shape[1]
                if wake_form == 'grid':
                    # wake_conn_iter = wake_connectivity[m][:i,:,:].reshape((i*nTp, 4))
                    wake_conn_iter = vlm_wake_connectivity[:i,:,:].reshape((-1, 4)) # OLD METHOD
                    # wake_conn_iter = wake_connectivity[m][-i:,:,:].reshape((-1, 4)) # NEW METHOD 
                    # NOTE: the line above doesn't work bc it's the point indices for the end of the wake, but we need the connectivity for the start of the wake to connect to the TE
                    reshaped_wake_points = np.reshape(wake_points_iter, (-1, 3))
                    vps = Mesh([reshaped_wake_points, wake_conn_iter], c='gray', alpha=1).linecolor('black')
                    # wake_color = np.reshape(wake_data_surf[:(i)*(nTp)], (-1,1)) # OLD METHOD
                    wake_color = np.reshape(wake_data_surf[-(i)*(nTp):], (-1,1)) # NEW METHOD
                    vps.cmap(cmap, wake_color, on='cells', vmin=min_mu, vmax=max_mu)

                    # vps.compute_normals(points=True)
                    # wake_normals = vps.vertex_normals
                    # asdf = reshaped_wake_points + wake_normals
                    # lines = Lines(reshaped_wake_points, asdf).linecolor('black')
                    # vp += lines

                elif wake_form == 'lines':
                    wpig = wake_points_iter.reshape((i+1, ns, 3))
                    # wdsg = wake_data_surf[:(i)*(nTp)].reshape((i, ns-1)) # OLD METHOD
                    wdsg = wake_data_surf[-(i)*(nTp):].reshape((i, ns-1)) # NEW METHOD
                    line_pts = []
                    line_colors = []
                    for j in range(i):
                        line_pts.extend([[wpig[j,ind,:], wpig[j+1,ind,:]] for ind in range(ns)])
                        
                        line_colors.append(wdsg[j,0])
                        line_colors.extend([(wdsg[j,ind]+wdsg[j,ind+1])/2. for ind in range(ns-2)])
                        line_colors.append(wdsg[j,-1])
                    vps = Lines(line_pts, lw=3, c='black')
                    vps.cmap(cmap, line_colors, on='cells', vmin=min_mu, vmax=max_mu)

                vp += vps
                vp += __doc__

                wns += num_surf_wake_nodes
                wps += num_surf_wake_panels
        
        if camera:
            plot_list = [vps]
            # plot_list.append(axs)
            show(plot_list, camera=camera, axes=False, interactive=interactive)
            # vp.show(axs, camera=camera, axes=False, interactive=interactive)  # render the scene
        elif side_view:
            vp.show(axs, elevation=-90, azimuth=0, roll=0,
                    axes=False, interactive=interactive)  # render the scene
        else:
            show([vps, axs], elevation=-45, azimuth=-45, roll=45,
                    axes=False, interactive=interactive)  # render the scene
        video.add_frame()  # add individual frame

    video.close()  # merge all the recorded frames



