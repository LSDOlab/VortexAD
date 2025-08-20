import numpy as np

def TE_detection(points, cells, edges2cells, threshold_theta=75, use_caddee=False):

    # getting edge vectors of panel
    # points are ordered s.t. outward normal
    p1_ind = cells[:,0]
    p2_ind = cells[:,1]
    p3_ind = cells[:,2]
    v1 = points[p2_ind,:] - points[p1_ind,:]
    v2 = points[p3_ind,:] - points[p2_ind,:]
    cell_normal = np.cross(v1, v2, axis=1)
    cell_normal_norm = np.linalg.norm(cell_normal, axis=1) # cell normal norm

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

    node_TE_indices = list(set(node_TE_indices))

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