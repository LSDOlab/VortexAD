import vedo
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show
import numpy as np
from vedo import *
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

def plot_wireframe(meshes, mesh_connectivity, wake_mesh, wake_connectivity, surface_data, wake_data, wake_form='grid', 
                   interactive=False, camera=False, surface_color='gray', cmap='jet', side_view=False, name='sample_gif', backend='imageio'):
    vedo.settings.default_backend = 'vtk'
    nt = surface_data.shape[0]
    num_meshes = len(meshes)
    axs = Axes(
        xrange=(0,3),
        yrange=(-7.5, 7.5),
        zrange=(0, 5),
    )
    video = Video(name+".mp4", fps=5, backend=backend)

    min_mu_b = np.min(surface_data)
    max_mu_b = np.max(surface_data)
    min_mu_w = np.min(wake_data)
    max_mu_w = np.max(wake_data)

    min_mu = np.min((min_mu_b, min_mu_w))
    max_mu = np.max((max_mu_b, max_mu_w))

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
        
        bps, bpe = 0, 0 # b panel start/end
        wns, wne = 0, 0 # wake node start/end
        wps, wpe = 0, 0 # wake panel start/end
        for m in range(num_meshes):
            print(f'mesh {m}')

            mesh_points = meshes[m][i,:]
            nc, ns = mesh_points.shape[0], mesh_points.shape[1]
            num_body_panels = (nc-1)*(ns-1)

            bpe += num_body_panels

            vps = Mesh([np.reshape(mesh_points, (-1, 3)), mesh_connectivity[m].reshape((-1,4))], c=surface_color, alpha=1.).linecolor('black')

            surf_color = np.reshape(surface_data[i,bps:bpe], (-1,1))
        
            vps.cmap(cmap, surf_color, on='cells', vmin=min_mu, vmax=max_mu)
            vps.add_scalarbar()
            vp += vps
            vp += __doc__

            if i > 0:
                num_surf_wake_nodes = nt*ns
                num_surf_wake_panels = (nt-1)*(ns-1)
                wne += num_surf_wake_nodes
                wpe += num_surf_wake_panels
                wake_mesh_surf = wake_mesh[i,wns:wne].copy().reshape((nt, ns, 3))
                wake_data_surf = wake_data[i,wps:wpe]
                
                wake_points_iter = wake_mesh_surf[:i+1,:]
                # wake_points_iter[0,:] = mesh_points[TE_indices]
                wake_points_iter[0,:] = mesh_points[-1,:]
                wake_points_iter = wake_points_iter.reshape((ns*(i+1), 3))

                nTp = wake_connectivity[m].shape[1]
                if wake_form == 'grid':
                    # wake_conn_iter = wake_connectivity[m][:i,:,:].reshape((i*nTp, 4))
                    wake_conn_iter = wake_connectivity[m][:i,:,:].reshape((-1, 4))
                    vps = Mesh([np.reshape(wake_points_iter, (-1, 3)), wake_conn_iter], c='gray', alpha=1).linecolor('black')
                    wake_color = np.reshape(wake_data_surf[:(i)*(nTp)], (-1,1))
                    vps.cmap(cmap, wake_color, on='cells', vmin=min_mu, vmax=max_mu)

                elif wake_form == 'lines':
                    wpig = wake_points_iter.reshape((i+1, ns, 3))
                    wdsg = wake_data_surf[:(i)*(nTp)].reshape((i, ns-1))
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

            bps += num_body_panels

        if camera:
            plot_list = [vps]
            plot_list.append(axs)
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