import numpy as np

def TE_detection(points, cells, edges2cells, points2cells, threshold_theta=75, 
                 edges2ignore=False, use_caddee=False):

    # getting edge vectors of panel
    # points are ordered s.t. outward normal

    cell_types = cells.keys()
    num_cells = np.sum([len(cells[cell_type]) for cell_type in cell_types])
    combined_cells = []
    for cell_type in cell_types:
        combined_cells += cells[cell_type].tolist()

    cell_normal = np.zeros((num_cells,3))
    cell_normal_norm = np.zeros((num_cells,))
    for i, cell_pts in enumerate(combined_cells):
        if len(cell_pts) == 3:
            v1 = points[cell_pts[1],:] - points[cell_pts[0],:]
            v2 = points[cell_pts[2],:] - points[cell_pts[1],:]
        elif len(cell_pts) == 4:
            v1 = points[cell_pts[2],:] - points[cell_pts[0],:]
            v2 = points[cell_pts[3],:] - points[cell_pts[1],:]
        cell_normal_iter = np.cross(v1, v2)
        cell_normal_norm_iter = np.linalg.norm(cell_normal_iter)
        cell_normal[i,:] = cell_normal_iter
        cell_normal_norm[i] = cell_normal_norm_iter
            
    # cosine of TE threshold angle
    theta_t = np.deg2rad(threshold_theta)
    threshold_cos = np.cos(theta_t)

    gcs_scaler = 1
    if use_caddee:
        gcs_scaler = -1

    '''
    The loop below finds the TE edges based on 3 criteria:
    - CRITERIA 1: norm of dot product between normals is below some threshold
        - sharpness of angle between two panels is the main criteria
    - CRITERIA 2: at least one of the vectors points downstream
        - NOTE: REFERENCE FRAMES MATTER HERE (thinking about CADDEE)
        - This leaves out leading edge panels
    - CRITERIA 3: flow turns away from surface
        - this will ignore crevices that "recirculate" flow into the body
        - we want the flow to turn away from the body for wake-shedding
    '''

    upper_TE_cells = []
    lower_TE_cells = []
    TE_edges = []
    node_TE_indices = []
    for edge in edges2cells.keys():
        cell_pairs = edges2cells[edge]
        if len(cell_pairs) < 2:
            continue
        cell_1, cell_2 = cell_pairs[0], cell_pairs[1]
        n1 = cell_normal[cell_1]/cell_normal_norm[cell_1]
        n2 = cell_normal[cell_2]/cell_normal_norm[cell_2]

        # CRITERIA 1
        edge_angle_cos = np.dot(n1, n2)
        if edge_angle_cos > threshold_cos:
            continue
        
        # CRITERIA 2:
        if gcs_scaler*n1[0] <= 0:
            if gcs_scaler*n2[0] <= 0:
                continue

        # CRITERIA 3:

        n_cross = np.cross(n1, n2)
        # c3 = np.dot(l, n_cross)
        # if c3 < 0:
        #     continue

        # CRITERIA 4: dealing with intersections between lifting surfaces and bodies (fuselage)
        '''
        main idea:
        - each TE node is attached to a set of panels
        - for each node on the edge, we get the cells that use that node
        - when compared to the cells of interest (cell_1, cell_2), the other cells need to point
            in a reasonably similar direction
            - another way of saying this: normalized cross product < threshold ~ 0.25 or so
        - this does not work well due to the wing tip mesh element orientation
        - for now, we will just add a manual criteria to ignore certain edges
        '''
        if False:
            pt_1, pt_2 = edge[0], edge[1]
            pt_1_cells = points2cells[pt_1]
            pt_2_cells = points2cells[pt_2]

            pt_1_cross_prod_cond = []
            pt_2_cross_prod_cond = []

            for cell in pt_1_cells:
                cell_pt_indices = combined_cells[cell]
                cell_pts = points[cell_pt_indices]
                ncp = len(cell_pts)
                if ncp == 3:
                    cross_prod = np.cross(cell_pts[1]-cell_pts[0], cell_pts[2]-cell_pts[1])
                elif ncp == 4:
                    cross_prod = np.cross(cell_pts[2]-cell_pts[0], cell_pts[3]-cell_pts[1])
                
                normal_vec = cross_prod / np.linalg.norm(cross_prod)

                asdf = np.linalg.norm(np.cross(n1, normal_vec))
                pt_1_cross_prod_cond.append(asdf)
        # print(edge)
        if edges2ignore:
            edge_listify = list(edge)
            edge_in_edges2ignore = False
            for val in edges2ignore:
                if edge_listify == val or edge_listify == val[::-1]:
                    edge_in_edges2ignore = True

            if edge_in_edges2ignore:
                # print(f'edge {edge} removed')
                continue
        
        # print(f'edge {edge} not removed')
        # print(edges2ignore)


        # finding upper and lower cells
        # upper: other vertex is above the edge
        #   - normal vector points up
        # lower: other vertex is below the edge
        #   - normal vector points down

        if n1[2] > 0:
            upper_TE_cells.append(int(cell_1))
            lower_TE_cells.append(int(cell_2))
        # elif n2[2] > 0:
        else:
            upper_TE_cells.append(int(cell_2))
            lower_TE_cells.append(int(cell_1))

        edge_pt_0 = points[edge[0],:]
        edge_pt_1 = points[edge[1],:]

        # if edge_pt_0[1] > edge_pt_1[1]-0.2:
        if edge_pt_0[1] > edge_pt_1[1]:
            TE_edges.append(edge[::-1])
            node_TE_indices.extend(edge[::-1])
        elif edge_pt_0[1] < edge_pt_1[1]:
            TE_edges.append(edge)
            node_TE_indices.extend(edge)
        else:
            TE_edges.append(edge)
            node_TE_indices.extend(edge)

        # NOTE: add a loop here that checks the ordering of the TE edges
        #   - we need to make sure that the node indices in the edge 
        #   preserve the proper ordering for the correct normal vector

    node_TE_indices = np.array(list(set(node_TE_indices)))

    return upper_TE_cells, lower_TE_cells, TE_edges, node_TE_indices




'''
TODO:
We need an even more general approach to computing trailing edges locations and edges


Steps:
- gather all of the trailing edge data like above (does not need to be clean in any way)
    - the code above can be used to figure out the TE elements and edges, but we need a 
        better way to deduce which element is UPPER and which is LOWER
- partition the trailing edges to figure out where the TE discontinuities occur
    - this could be between wing, tail, rotors, etc.
    - we can do this by looping through the edge indices and figuring out where edges are
        connected by looking at node indices, etc. (could we use a KDTree or a tree of some kind?)
- we can then loop from the -y to +y direction along each subdivision of the trailing edges to 
    figure out which trailing edge surface is upper or lower
    - we can use a greedy nearest neighbor walk to traverse from one end to the other
    - still unsure about how to determine the ordering; options are:
        - look at relative OOM of normal vector; upper surface will likely have more of a 
            component in the streamwise direction
            - can't always order from -y to +y bc upper and lower more or less refer to whichever
                sides refer to suction or pressure. 
'''