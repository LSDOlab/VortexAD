import numpy as np

def find_cell_adjacency_old(points, cells, radius=1.e-10):
    '''
    points is a numpy array of shape (num_points, 3) where the 3 is the x,y,z coordinates
    cells is a numpy array of shape (num_cells, 3) where the 3 is the node indices of the cell
    '''

    # Finding and REMOVING duplicate nodes within a given radius
    points, cells = remove_duplicate_nodes(points, cells, radius)

    num_pts = len(points)
    num_cells = len(cells)

    # Finding the edge corresponding to a cell (edge tuple is the dictionary key)
    edges2cells = {}
    for c, cell in enumerate(cells):
        cell = list(cell)
        cell.append(cell[0])
        edges = [(cell[i], cell[i+1]) for i in range(len(cell)-1)]

        for edge in edges:
            edge_rev = edge[::-1]
            if edge in edges2cells.keys():
                edges2cells[edge].append(c)
            elif edge_rev in edges2cells.keys():
                edges2cells[edge_rev].append(c)
            else:
                edges2cells[edge] = [c]
        1

    # return edges2cells

    # Finding neighboring cells (cell index is the dict key)
    cell_adjacency = {i: [] for i in range(num_cells)}
    for edge in edges2cells.keys():
        cell_pairs = edges2cells[edge]
        if len(cell_pairs) < 2:
            continue # means this is an edge on the border of the mesh
        cell_adjacency[cell_pairs[0]].append(cell_pairs[1])
        cell_adjacency[cell_pairs[1]].append(cell_pairs[0])

    cell_adjacency = np.array(list(cell_adjacency.values()), dtype=int)
    # # # # cell_adjacency = list(cell_adjacency.values())

    # Finding elements corresponding to each node/vertex
    points2cells = {i: [] for i in range(num_pts)}
    for c, cell in enumerate(cells):
        for ind in cell:
            points2cells[ind].append(c)

    return points, cells, cell_adjacency, edges2cells, points2cells

def remove_duplicate_nodes(points, cells, radius=1.e-10):
    num_pts_orig = len(points)

    npu = 0 # number of unique points
    while True:

        num_pts_in_loop = len(points)
        if npu == num_pts_in_loop:
            break

        # ==== finding indices where point is duplicated ====
        pt_loop = points[npu,:] # point in list
        pt_delta = np.linalg.norm(points[npu+1:,:] - pt_loop, axis=1) # delta norm of point and remaining points
        # pt_delta = np.linalg.norm(points - pt_loop, axis=1) # delta norm of point and remaining points
        dup_ind_list = list(np.where(pt_delta < radius)[0]+npu+1) # list where duplicates exist

        if not dup_ind_list: # if no duplicates, go to next iteration in loop
            npu += 1
            continue
        print(dup_ind_list)
        print(f'removing duplicate of point {npu}')
        # ==== adjusting point array to remove duplicates ====
        points = np.delete(points, dup_ind_list, axis=0) # adjusting points

        # ==== adjusting cells to remove duplicate point indices ====
        for dup_ind in dup_ind_list:
            asdf = np.where(cells == dup_ind)
            row_ind, col_ind = list(asdf[0]), list(asdf[1])
            cells[row_ind, col_ind] = npu

        # ==== adjusting cells for shifted points
        shift_ind = 0
        for dup_ind in dup_ind_list:
            asdf = np.where(cells > dup_ind - shift_ind)
            row_ind, col_ind = list(asdf[0]), list(asdf[1])
            cells[row_ind, col_ind] += -1
            shift_ind += 1
        # break
        npu += 1 # moving index for point

        
    return points, cells