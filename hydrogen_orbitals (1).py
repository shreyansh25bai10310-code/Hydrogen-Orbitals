
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Physical Constants & Setup ---
a0 = 1.0  # Bohr radius (atomic units)

# --- 2. Coordinate Systems ---
def get_grid(extent=20, resolution=400, z_slice=0.0):
    """
    Generates a 2D grid in the xy-plane at a specific z-height.
    """
    x = np.linspace(-extent, extent, resolution)
    y = np.linspace(-extent, extent, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_slice)
    
    # Convert to spherical coordinates (r, theta, phi)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Handle division by zero for Theta at origin
    with np.errstate(divide='ignore', invalid='ignore'):
        Theta = np.arccos(Z / R)
        Theta = np.nan_to_num(Theta, nan=0.0) 

    Phi = np.arctan2(Y, X)
    
    return X, Y, R, Theta, Phi

# --- 3. Wavefunctions (Real Realizations) ---

def psi_1s(r, theta, phi):
    # n=1, l=0, m=0
    return (1/np.sqrt(np.pi)) * np.exp(-r)

# --- NEW: 2s Orbital ---
def psi_2s(r, theta, phi):
    # n=2, l=0, m=0
    # Has 1 radial node
    prefactor = 1.0 / (4.0 * np.sqrt(2.0 * np.pi))
    radial = (2.0 - r) * np.exp(-r/2.0)
    return prefactor * radial

def psi_2pz(r, theta, phi):
    # n=2, l=1, m=0 (Aligned along Z)
    rho = r / 2.0
    radial = (1.0 / (np.sqrt(32 * np.pi))) * r * np.exp(-rho)
    angular = np.cos(theta)
    return radial * angular

def psi_2px(r, theta, phi):
    # n=2, l=1, m=+/-1 (Real superposition, Aligned along X)
    rho = r / 2.0
    prefactor = (1.0 / (np.sqrt(32 * np.pi)))
    return prefactor * r * np.exp(-rho) * np.sin(theta) * np.cos(phi)

def psi_2py(r, theta, phi):
    # n=2, l=1, m=+/-1 (Real superposition, Aligned along Y)
    rho = r / 2.0
    prefactor = (1.0 / (np.sqrt(32 * np.pi)))
    return prefactor * r * np.exp(-rho) * np.sin(theta) * np.sin(phi)

# --- NEW: 3s Orbital ---
def psi_3s(r, theta, phi):
    # n=3, l=0, m=0
    # Has 2 radial nodes
    prefactor = 1.0 / (81.0 * np.sqrt(3.0 * np.pi))
    radial = (27.0 - 18.0*r + 2.0*(r**2)) * np.exp(-r/3.0)
    return prefactor * radial

# --- NEW: 3p Orbitals ---
def psi_3pz(r, theta, phi):
    # n=3, l=1, m=0 (Z-aligned)
    # Has 1 radial node (unlike 2p which has 0)
    prefactor = np.sqrt(2.0) / (81.0 * np.sqrt(np.pi))
    radial = (6.0*r - r**2) * np.exp(-r/3.0)
    angular = np.cos(theta)
    return prefactor * radial * angular

def psi_3px(r, theta, phi):
    # n=3, l=1 (X-aligned)
    prefactor = np.sqrt(2.0) / (81.0 * np.sqrt(np.pi))
    radial = (6.0*r - r**2) * np.exp(-r/3.0)
    angular = np.sin(theta) * np.cos(phi)
    return prefactor * radial * angular

def psi_3py(r, theta, phi):
    # n=3, l=1 (Y-aligned)
    prefactor = np.sqrt(2.0) / (81.0 * np.sqrt(np.pi))
    radial = (6.0*r - r**2) * np.exp(-r/3.0)
    angular = np.sin(theta) * np.sin(phi)
    return prefactor * radial * angular

def psi_3dz2(r, theta, phi):
    # n=3, l=2, m=0
    prefactor = 1.0 / (81.0 * np.sqrt(6.0 * np.pi))
    radial_part = (r**2) * np.exp(-r/3.0)
    angular_part = (3 * np.cos(theta)**2 - 1)
    return prefactor * radial_part * angular_part

def psi_3dxz(r, theta, phi):
    # n=3, l=2
    prefactor = 1.0 / (81.0 * np.sqrt(2.0 * np.pi))
    radial_part = (r**2) * np.exp(-r/3.0)
    angular_part = np.sin(theta) * np.cos(theta) * np.cos(phi)
    return prefactor * radial_part * angular_part

# Map strings to functions
ORBITAL_MAP = {
    "1s": psi_1s,
    "2s": psi_2s,     # Added
    "2pz": psi_2pz,
    "2px": psi_2px,
    "2py": psi_2py,
    "3s": psi_3s,     # Added
    "3pz": psi_3pz,   # Added
    "3px": psi_3px,   # Added
    "3py": psi_3py,   # Added
    "3dz2": psi_3dz2,
    "3dxz": psi_3dxz
}

# --- 4. Main Plotting Logic ---

def plot_orbital(orbital_name, extent, z_slice):
    if orbital_name not in ORBITAL_MAP:
        print(f"Error: {orbital_name} not found.")
        return

    print(f"\n> Calculating {orbital_name} orbital at z = {z_slice} a.u...")
    
    # 1. Get Grid
    X, Y, R, Theta, Phi = get_grid(extent=extent, resolution=500, z_slice=z_slice)
    
    # 2. Calculate Wavefunction (Psi)
    func = ORBITAL_MAP[orbital_name]
    psi = func(R, Theta, Phi)
    
    # 3. Calculate Probability Density (Psi^2)
    prob_density = psi**2
    
    # 4. Plot
    plt.figure(figsize=(8, 7))
    plt.imshow(prob_density, extent=[-extent, extent, -extent, extent], 
               origin='lower', cmap='inferno', interpolation='bilinear')
    
    plt.colorbar(label='Probability Density $|\psi|^2$')
    plt.title(f"Hydrogen Orbital: {orbital_name}\nSlice: z = {z_slice} $a_0$")
    plt.xlabel(f"x ($a_0$)")
    plt.ylabel(f"y ($a_0$)")
    
    # Contour lines
    max_val = np.max(prob_density)
    if max_val > 0:
        levels = np.linspace(max_val*0.01, max_val, 7)
        plt.contour(X, Y, prob_density, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
    
    print("> Plot generated. Close the window to continue.")
    plt.show()

# --- 5. User Input Loop ---

def main():
    print("--- Hydrogen Orbital Visualizer ---")
    print("This tool plots the probability density of electron orbitals.")
    
    while True:
        print("\n" + "="*40)
        # Sort keys to make the list look organized
        keys = sorted(ORBITAL_MAP.keys())
        print(f"Available Orbitals: {', '.join(keys)}")
        
        # 1. Get Orbital Name
        user_choice = input("Enter orbital name (or 'q' to quit): ").strip()
        
        if user_choice.lower() in ['q', 'quit', 'exit']:
            print("Exiting...")
            break
            
        if user_choice not in ORBITAL_MAP:
            print(f"Invalid orbital '{user_choice}'. Please try again.")
            continue

        # 2. Get Extent and Z-slice with error handling
        try:
            # Provide different defaults based on orbital size for better UX
            default_extent = 30.0 if '3' in user_choice else 15.0
            
            extent_input = input(f"Enter plot range [default: {default_extent}]: ")
            extent = float(extent_input) if extent_input else default_extent
            
            z_input = input("Enter Z-axis slice position [default: 0.0]: ")
            z_slice = float(z_input) if z_input else 0.0
            
            # Run the plotter
            plot_orbital(user_choice, extent, z_slice)
            
        except ValueError:
            print("Error: Please enter valid numbers for range and z-slice.")

if __name__ == "__main__":
    main()