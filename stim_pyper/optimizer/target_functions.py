import numpy as np

# NOTE: During the assign values process, we have access to the contact's coords and get the ultimate mask. calculation 
# NOTE: Since we have the paired [x,y,z] coordinates of each index in the mask from L[:, :3], we can find distance
# NOTE: from each contact's coords. This distance can be used to convert mask-values to electrical field values 
# NOTE: using a kernel which takes K([xyz]_i, [xyz]_c) and computes e = k (q.r**2)

# -----------------------------
# Target and Penalty Functions
# -----------------------------

def get_q(r, k, V=0.1):
    '''
    Field at most peripheral stimulation volume is 0.1V
    Use this to find charge at origin of stimulation volume. 
    '''
    return (V * r**2) / k

def electric_potential_kernel(L, coord, r, k=8.99e9, e=1e-8):
    '''
    Generate a spherical electrical potential around a point source (contact). Take the gradient of this to get the e-field. 
    Args:
        L: (N, 4) array, where cols 0:3 are x,y,z coords
        coord: (3,) contact location
        r: radius from contact, used to estimate total charge
        k: Coulomb constant (default 8.99e9 N·m²/C²)
        e: epsilon to avoid divide-by-zero
        
    Returns:
        np.array of electrical field in N/C
    '''
    k_mm = k * 1000**2                                                      # Convert K N·m²/C² → N·mm²/C²
    q = get_q(r, k_mm)                                                      # Get Coulombs of origin of stimulation
    pairwise_dist = np.linalg.norm(np.array(L[:, :3]) - np.array(coord), axis=1)    # find pairwise distance from the coordinate to all points
    return k_mm * (q / (pairwise_dist**2 + e) )                                       # return scalar field in 

def compute_radius(milliamps):
    """
    Compute the radius based on the input current in milliamps.
    """
    radius = (milliamps - 0.1) / 0.22
    return np.where(milliamps > 0.1, np.sqrt(radius), 0)

def assign_sphere_values(L, center, r):
    """
    Assign binary values to points in L based on whether they are inside a sphere.
    
    Args:
        L: Array of points with coordinates.
        center: Center of the sphere (x0, y0, z0).
        r: Radius of the sphere.
    
    Returns:
        Binary array indicating points inside the sphere.
    """
    x0, y0, z0 = center
    x_coords = L[:, 0]
    y_coords = L[:, 1]
    z_coords = L[:, 2]
    dx = x_coords - x0
    dy = y_coords - y0
    dz = z_coords - z0
    distance_squared = dx**2 + dy**2 + dz**2
    inside_sphere = distance_squared < r**2
    return inside_sphere.astype(int)

def assign_directional_values(L, directional_model):
    """
    Assign binary values to points in L based on directional VTA boundaries.
    
    Args:
        L: Array of points with coordinates.
        directional_model: an instance of EvaluateDirectionalVta
    
    Returns:
        Binary array indicating valid VTA points (1 if inside, 0 if outside).
    """
    coords = L[:, :3]
    mask = directional_model.evaluate_planar_and_spherical_points(coords)
    return mask.astype(int)

def get_sphere_volume(r):
    return 4/3 * r**3 * np.pi

def dot_product(S, L):
    """
    Compute the dot product of the binary sphere array and magnitudes in L.
    
    Args:
        S: Binary array indicating points inside the sphere.
        L: Array of points with magnitudes.
    
    Returns:
        Dot product result.
    """
    magnitudes = L[:, 3]
    return np.dot(S, magnitudes)

def target_function(S_r, L, volume, e=1e-8):
    """
    Compute the target function value for a sphere. 
    Result is total density of voxel values inside sphere.
    
    Args:
        S_r: Binary array indicating points inside the sphere.
        L: Array of points with magnitudes.
        r: Radius of the sphere.
        e: Prevents division by zero. 
    
    Returns:
        Target function value.
    """
    numerator = dot_product(S_r, L)
    return numerator / (volume + e)

def generate_V(L, sphere_coords, v, directional_models=None):
    '''
    Computes the overall mask assocaited with each coordinate, v, and directionality. 
    Args:
        L: Array or points with magnitudes
        sphere_coords: List of sphere center coordiantes
        v: Array of current values for each sphere.
        directional_models: Optional list of EvaluateDirectionalVta instances (or None).
    Returns:
        V: Array of mask for all VTAs
    '''
    r = compute_radius(v)                   # broadcast r calculation
    V = np.zeros(L.shape[0], dtype=bool)    # initialize array to place masks into
    for index in range(len(sphere_coords)):
        if directional_models and directional_models[index]:
            directional_models[index].radius_mm = r[index]
            V_i = assign_directional_values(L, directional_models[index])
        else:
            V_i = assign_sphere_values(L, sphere_coords[index], r[index])
        V |= V_i.astype(bool)               # inplace union
    return V

def target_function_handler(sphere_coords, v, L, directional_models=None):
    """
    Compute the total target function value for all spheres or directional VTAs.
    
    Args:
        sphere_coords: List of sphere centers.
        v: Array of current values for each sphere.
        L: Array of points with magnitudes.
        directional_models: Optional list of EvaluateDirectionalVta instances (or None).
    
    Returns:
        Sum of target function values for all VTAs.
    """
    sum_target_value = 0
    for index, center in enumerate(sphere_coords):
        if v[index] > 0.1:
            r = compute_radius(v[index])
            volume = get_sphere_volume(r)
            if directional_models and directional_models[index]:
                directional_models[index].radius_mm = r
                S_r = assign_directional_values(L, directional_models[index])
                volume = volume / 3
            else:
                S_r = assign_sphere_values(L, center, r)
            target_value = target_function(S_r, L, volume)
            sum_target_value += target_value
    return sum_target_value

def target_function_handlerV2(sphere_coords, v, L, directional_models=None):
    """
    Compute the total target function value for all spheres or directional VTAs.
    
    Args:
        sphere_coords: List of sphere centers.
        v: Array of current values for each sphere.
        L: Array of points with magnitudes.
        directional_models: Optional list of EvaluateDirectionalVta instances (or None).
    
    Returns:
        Sum of target function values for all VTAs.
    """
    V = generate_V(L, sphere_coords, v, directional_models)
    volume = np.sum(V)
    return target_function(V, L, volume)

def penalty_per_contact(v_i, lam):
    """
    Compute the penalty for a single contact based on current and lambda.
    
    Args:
        v_i: Current value for the contact.
        lam: Penalty scaling factor.
    
    Returns:
        Penalty value for the contact.
    """
    if v_i > 0.1:
        return lam * v_i
    return 0

def penalty_per_contact_handler(v, lam):
    """
    Compute the total penalty for all contacts.
    
    Args:
        v: Array of current values for all contacts.
        lam: Penalty scaling factor.
    
    Returns:
        Total penalty value for all contacts.
    """
    return sum(penalty_per_contact(val, lam) for val in v)

def penalty_all_contacts(v, lam):
    """
    Compute the penalty for all contacts based on the total current.
    
    Args:
        v: Array of current values for all contacts.
        lam: Penalty scaling factor.
    
    Returns:
        Penalty value for all contacts.
    """
    if sum(v) > 0.1:
        return lam / (5 - sum(v))
    return 0

def loss_function(sphere_coords, v, L, lam, directional_models=None):
    """
    Compute the loss function as the difference between target and penalties.

    Args:
        sphere_coords: List of sphere centers.
        v: Array of current values for each sphere.
        L: Array of points with magnitudes in last column (N_Voxels, [x,y,z,magnitude]).
        lam: Penalty scaling factor.
        directional_models: Optional list of EvaluateDirectionalVta instances (or None).

    Returns:
        Loss function value.
    """
    T = target_function_handlerV2(sphere_coords, v, L, directional_models)
    P1 = penalty_per_contact_handler(v, lam)
    P2 = penalty_all_contacts(v, lam)
    return T - P1 - P2

