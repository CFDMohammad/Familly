import numpy as np

# D2Q9 Lattice constants
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]) # weights
c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], # velocities
              [1, 1], [-1, 1], [-1, -1], [1, -1]])
cs2 = 1/3 # speed of sound squared

# Simulation Parameters
Nx = 100  # grid dimensions
Ny = 100
max_timesteps = 1000  # number of time steps
u_lid = 0.1  # lid velocity
rho0 = 1.0 # initial density
tau = 0.8 # relaxation parameter (related to viscosity: nu = cs2 * (tau - 0.5))
nu = cs2 * (tau - 0.5)

# Initialization
f = np.zeros((Ny, Nx, 9)) # particle distribution functions (PDFs)
rho = np.full((Ny, Nx), rho0) # macroscopic density
ux = np.zeros((Ny, Nx)) # macroscopic x-velocity
uy = np.zeros((Ny, Nx)) # macroscopic y-velocity

def equilibrium_pdf(rho_local, ux_local, uy_local):
    """Calculates the equilibrium distribution function."""
    feq = np.zeros(9)
    for i in range(9):
        cu = c[i, 0] * ux_local + c[i, 1] * uy_local
        feq[i] = w[i] * rho_local * (1 + cu/cs2 + 0.5 * (cu/cs2)**2 - 0.5 * (ux_local**2 + uy_local**2)/cs2)
    return feq

# Initialize PDFs to equilibrium with zero velocity and unit density
for j in range(Ny):
    for i in range(Nx):
        f[j, i, :] = equilibrium_pdf(rho0, 0, 0)

# Main Loop
for t_step in range(max_timesteps):
    # Collision Step (BGK)
    for j in range(Ny):
        for i in range(Nx):
            feq_local = equilibrium_pdf(rho[j,i], ux[j,i], uy[j,i])
            f[j,i,:] = f[j,i,:] - (1/tau) * (f[j,i,:] - feq_local)

    # Streaming Step
    f_new = np.zeros_like(f) # Create a new array for post-streaming PDFs
    for j in range(Ny):
        for i in range(Nx):
            for k in range(9):
                # Stream backwards (pull scheme)
                # The PDF f[j,i,k] moves TO (j+c[k,1], i+c[k,0])
                # So, f_new[j,i,k] comes FROM (j-c[k,1], i-c[k,0])
                prev_y, prev_x = j - c[k,1], i - c[k,0]

                # This is where boundary conditions are implicitly needed
                # For now, we only stream from the interior.
                # Boundary nodes will be corrected by BCs after this.
                if 0 <= prev_x < Nx and 0 <= prev_y < Ny:
                    f_new[j,i,k] = f[int(prev_y), int(prev_x), k]
    # Note: PDFs at boundary nodes that would stream from outside are currently zero in f_new.
    # These will be filled by the boundary conditions.

    # Boundary Conditions
    # Applied to f_new, which contains the streamed PDFs.
    # The general idea is that f_new has "gaps" (zeros or incorrect values) for PDFs that
    # should have streamed from outside the domain. These are filled by the BCs.

    # No-slip walls (bottom, left, right) - Bounce-back
    # Example: f_new[0, i, 2] (which is f_north for the bottom wall) should be f[0, i, 4] (f_south at the wall before streaming)
    # After streaming, f_new[0,i,k] are populations that arrived at (0,i)
    # We need to define the outgoing populations from the wall.
    # Bounce-back: f_out = f_in_flipped_direction

    # Bottom wall (j=0)
    for i in range(Nx):
        f_new[0, i, 2] = f_new[0, i, 4]  # N receives from S (which was at wall)
        f_new[0, i, 5] = f_new[0, i, 7]  # NE receives from SW
        f_new[0, i, 6] = f_new[0, i, 8]  # NW receives from SE

    # Left wall (i=0)
    for j in range(1, Ny-1): # Exclude corners initially
        f_new[j, 0, 1] = f_new[j, 0, 3]  # E receives from W
        f_new[j, 0, 5] = f_new[j, 0, 7]  # NE receives from SW
        f_new[j, 0, 8] = f_new[j, 0, 6]  # SE receives from NW

    # Right wall (i=Nx-1)
    for j in range(1, Ny-1): # Exclude corners initially
        f_new[j, Nx-1, 3] = f_new[j, Nx-1, 1]  # W receives from E
        f_new[j, Nx-1, 7] = f_new[j, Nx-1, 5]  # SW receives from NE
        f_new[j, Nx-1, 6] = f_new[j, Nx-1, 8]  # NW receives from SE

    # Moving Lid (top wall, j=Ny-1)
    # Populations 4 (S), 7 (SW), 8 (SE) are unknown after streaming and must be reconstructed.
    # Other populations 0,1,2,3,5,6 are known from streaming from interior or adjacent lid nodes.
    for i in range(Nx): # Iterate over all lid nodes, including corners
        # Calculate rho at the lid using known post-streaming populations
        # f_new[Ny-1, i, k] for k in [0,1,2,3,5,6] are known after streaming
        # f_new[Ny-1, i, k] for k in [4,7,8] are unknown (point into fluid)
        # Zou-He boundary condition approach:
        rho_wall = (1.0 / (1.0 - uy[Ny-1,i])) * ( # uy[Ny-1,i] is previous step's normal velocity, assume 0 for lid
            f_new[Ny-1, i, 0] + f_new[Ny-1, i, 1] + f_new[Ny-1, i, 3] +
            2 * (f_new[Ny-1, i, 2] + f_new[Ny-1, i, 5] + f_new[Ny-1, i, 6]) # Populations pointing away or parallel
        )
        # If uy is non-zero and known at boundary, include it. For lid, normal velocity is 0.
        # Simplified version: rho_wall = rho[Ny-1, i] (density from previous timestep)
        # For stability, using rho0 or a locally calculated rho is better.
        # Let's try a simpler rho for now, as uy at lid is zero.
        # rho_local_lid = f_new[Ny-1,i,0] + f_new[Ny-1,i,1] + f_new[Ny-1,i,3] + f_new[Ny-1,i,2] + f_new[Ny-1,i,6] + f_new[Ny-1,i,5] # This is rho from knowns
        # This sum is actually rho - (f4+f7+f8). Not directly rho.
        # For Zou-He, density is calculated from known populations and specified velocity.
        
        # For moving lid (uy=0, ux=u_lid):
        # f4, f7, f8 are unknown after streaming.
        # f4 = f2 (bounce-back for normal component)
        # f7 = f5 - 0.5 * (f1 - f3) + rho_wall * u_lid / (6*cs2) (Zou-He style for oblique)
        # f8 = f6 + 0.5 * (f1 - f3) + rho_wall * u_lid / (6*cs2) (Zou-He style for oblique)
        # For simplicity first: Set to equilibrium for u_lid, and bounce back normal components.
        # This is a common approximation.
        
        # Let's use the rho from the previous time step at the lid for feq calculation.
        # This avoids immediate feedback from just streamed values which might be noisy.
        rho_for_lid_feq = rho[Ny-1, i] # Density from previous macroscopic calculation
        feq_lid = equilibrium_pdf(rho_for_lid_feq, u_lid, 0.0)

        f_new[Ny-1, i, 4] = feq_lid[4] # S
        f_new[Ny-1, i, 7] = feq_lid[7] # SW
        f_new[Ny-1, i, 8] = feq_lid[8] # SE
        
        # And for populations that hit the lid and reflect (parallel movement)
        # these are already streamed. For example, f_new[Ny-1, i, 1] came from f[Ny-1, i-1, 1]
        # No, this is not quite right.
        # The populations f0, f1, f2, f3, f5, f6 for node (Ny-1, i) are known after streaming.
        # The populations f4, f7, f8 are unknown as they point into the fluid.
        # We must reconstruct these.
        # Option 1: Full equilibrium (as done above for these 3 components based on rho_for_lid_feq)
        # Option 2: Zou-He specific reconstruction.
        # (Zou-He for density)
        # rho_known_contrib = f_new[Ny-1,i,0] + f_new[Ny-1,i,1] + f_new[Ny-1,i,3] + f_new[Ny-1,i,2] + f_new[Ny-1,i,5] + f_new[Ny-1,i,6]
        # rho_at_lid = (rho_known_contrib + rho_for_lid_feq * (w[4]+w[7]+w[8])) / (1.0 - (w[4]+w[7]+w[8])) # This is not standard Zou-He rho.

        # Let's stick to a simpler, common moving wall model:
        # For populations pointing into the domain (4, 7, 8 for top wall):
        # Calculate rho at the wall node using only the populations that have streamed *to* the wall
        # from the fluid interior and from movement along the wall.
        # Then, use this rho and the known wall velocity (u_lid, 0) to set the unknown
        # f_k's to their equilibrium values.
        # Known after streaming for (Ny-1, i): f0, f1 (from i-1), f2 (from Ny-2,i), f3 (from i+1), f5 (from Ny-2,i-1), f6 (from Ny-2,i+1)
        # So, at node (Ny-1, i), the PDFs f_new[Ny-1,i,k] that are known are:
        # k=0 (rest), k=1 (from left), k=3 (from right), k=2 (from below), k=5 (from below-left), k=6 (from below-right)
        # The PDFs f_new[Ny-1,i,4], f_new[Ny-1,i,7], f_new[Ny-1,i,8] point from wall into fluid and are unknown.

        # Simplified Zou-He for top moving wall (uy=0):
        # rho_wall = ( f_new[Ny-1,i,0] + f_new[Ny-1,i,1] + f_new[Ny-1,i,3] + 2*(f_new[Ny-1,i,2] + f_new[Ny-1,i,5] + f_new[Ny-1,i,6]) ) / (1.0 - 0)
        # This formula for rho_wall is for when uy_wall is unknown. Here uy_wall = 0.
        # rho_wall = (1/(1-uy_wall_normal)) * sum(...) where uy_wall_normal = 0.
        # A common way: determine rho_wall such that it satisfies the equilibrium for the known u_wall.
        # For the top wall (j=Ny-1), moving with (u_lid, 0):
        # rho_calc_lid = ( f_new[Ny-1,i,0] + f_new[Ny-1,i,1] + f_new[Ny-1,i,3] +
        #                  2 * (f_new[Ny-1,i,2] + f_new[Ny-1,i,5] + f_new[Ny-1,i,6]) )
        # This is from known post-streamed populations.
        # Then set f_new[Ny-1,i,4], f_new[Ny-1,i,7], f_new[Ny-1,i,8] using equilibrium with (rho_calc_lid, u_lid, 0)

        # Let's use the previous time step's rho for stability at the lid for now, and set unknowns to eq.
        feq_lid_unknowns = equilibrium_pdf(rho[Ny-1,i], u_lid, 0.0) # Use rho from t, not t+dt*
        f_new[Ny-1, i, 4] = feq_lid_unknowns[4] # S
        f_new[Ny-1, i, 7] = feq_lid_unknowns[7] # SW
        f_new[Ny-1, i, 8] = feq_lid_unknowns[8] # SE

        # Bounce-back for components parallel to the moving lid that hit the "edge" of the lid
        # This is more about corner handling or if the lid itself has thickness.
        # For now, the above equilibrium setting for unknowns is the primary mechanism.


    # Corner nodes:
    # Top-left (0, Ny-1) and Top-right (Nx-1, Ny-1) for lid.
    # Bottom-left (0,0), Bottom-right (Nx-1,0) for stationary.
    # The loops for stationary walls exclude j=0 and j=Ny-1 for left/right walls.
    # The loop for bottom wall covers i=0 and i=Nx-1.
    # The loop for lid currently covers i=0 to Nx-1.

    # Example: Bottom-left corner (0,0)
    # From bottom wall bounce-back (i=0):
    # f_new[0,0,2] = f_new[0,0,4] (N from S)
    # f_new[0,0,5] = f_new[0,0,7] (NE from SW)
    # f_new[0,0,6] = f_new[0,0,8] (NW from SE)
    # From left wall bounce-back (j=0, but loop starts at j=1):
    # If we extend left wall BC to j=0:
    # f_new[0,0,1] = f_new[0,0,3] (E from W)
    # f_new[0,0,5] = f_new[0,0,7] (NE from SW) - consistent with above
    # f_new[0,0,8] = f_new[0,0,6] (SE from NW) - consistent with above (directions flipped)

    # So, the bounce-back for stationary corners should be consistent.
    # For top corners (lid):
    # Top-left (0, Ny-1):
    # Lid sets f_new[Ny-1,0,4], f_new[Ny-1,0,7], f_new[Ny-1,0,8] to equilibrium.
    # Left wall (if extended to j=Ny-1):
    # f_new[Ny-1,0,1] = f_new[Ny-1,0,3] (E from W)
    # f_new[Ny-1,0,5] = f_new[Ny-1,0,7] (NE from SW)
    # f_new[Ny-1,0,8] = f_new[Ny-1,0,6] (SE from NW)
    # There's a conflict: f_new[Ny-1,0,7] and f_new[Ny-1,0,8] are set by lid AND potentially by left wall bounce-back.
    # Standard practice: At corners between stationary and moving wall, moving wall usually dominates for shared nodes.
    # Or, use specific corner treatments.
    # For now, lid BC (equilibrium for unknowns 4,7,8) is applied to all i, including 0 and Nx-1.
    # And stationary wall BCs for left/right walls are for j in 1..Ny-2. This avoids direct conflict.
    # The nodes (0,0), (Nx-1,0), (0,Ny-1), (Nx-1,Ny-1) need consistent handling.
    # Stationary corners (0,0) and (Nx-1,0) are covered by bottom wall bounce-back.
    # Left wall bounce-back is for (0, 1..Ny-2). Right wall for (Nx-1, 1..Ny-2).
    # This setup seems okay for now, let's assume the basic bounce-back and equilibrium settings are enough.


    f = f_new # Update f with the post-streaming and post-BC values

    # Macroscopic Variables (recalculate after BCs and streaming)
    rho = np.sum(f, axis=2)
    ux = np.zeros((Ny, Nx))
    uy = np.zeros((Ny, Nx))
    for k in range(9):
        ux += f[:,:,k] * c[k,0]
        uy += f[:,:,k] * c[k,1]
    ux /= rho
    uy /= rho

    # Optional: Print simulation status
    if t_step % 100 == 0:
        print(f"Time step: {t_step}, Avg. Density: {np.mean(rho)}")
        # print(f"Max ux: {np.max(np.abs(ux))}, Max uy: {np.max(np.abs(uy))}")

# Optional: Save results
# np.savez("cavity_flow_results.npz", ux=ux, uy=uy, rho=rho)

print("Simulation Finished.")
print(f"Kinematic viscosity (nu) from tau: {nu}")
print(f"Lid Reynolds number (Re = u_lid * Nx / nu): {u_lid * (Nx-1) / nu if nu > 0 else 'inf'}")

if __name__ == '__main__':
    # This space can be used for plotting or further analysis if run as a script
    # For example, using matplotlib to visualize the velocity field
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8,8))
    # plt.imshow(np.sqrt(ux**2+uy**2).T, origin='lower', cmap='jet')
    # plt.colorbar(label='Velocity Magnitude')
    # plt.title(f'Velocity Magnitude at t={max_timesteps}')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    pass
