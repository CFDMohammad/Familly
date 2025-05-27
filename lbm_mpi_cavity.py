import numpy as np
from mpi4py import MPI # Import MPI

# D2Q9 Lattice constants
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]) # weights
c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], # velocities
              [1, 1], [-1, 1], [-1, -1], [1, -1]])
cs2 = 1/3 # speed of sound squared

# Simulation Parameters (Global)
Nx_global = 129  # grid dimensions (for 128x128 cells)
Ny_global = 129
max_timesteps = 20000  # number of time steps for Re=1000
u_lid = 0.1  # lid velocity
rho0 = 1.0 # initial density
# For Re=1000, L = Nx_global-1 = 128. u_lid = 0.1
# nu_target = u_lid * L / Re = 0.1 * 128 / 1000 = 0.0128
# tau = (nu_target / cs2) + 0.5 = (0.0128 / (1/3)) + 0.5 = 0.0384 + 0.5 = 0.5384
tau = 0.5384 # relaxation parameter
nu = cs2 * (tau - 0.5) # Effective kinematic viscosity

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Domain Decomposition (1D along Y-axis)
# Calculate local Ny and starting row for each process
# Basic division
rows_per_process = Ny_global // size
remainder_rows = Ny_global % size

local_Ny = rows_per_process + (1 if rank < remainder_rows else 0)
start_row_global = rank * rows_per_process + min(rank, remainder_rows)
# end_row_global = start_row_global + local_Ny - 1 # For reference

# Local grid dimensions (Nx is not decomposed)
Nx = Nx_global

# Initialization
# Add 2 for halo layers (1 top, 1 bottom)
f = np.zeros((local_Ny + 2, Nx, 9)) # particle distribution functions (PDFs)
rho = np.full((local_Ny + 2, Nx), rho0) # macroscopic density, including halos for convenience
ux = np.zeros((local_Ny + 2, Nx)) # macroscopic x-velocity
uy = np.zeros((local_Ny + 2, Nx)) # macroscopic y-velocity

# Corrected local f and macroscopic variables initialization (only for interior points)
# Halos will be filled by communication or boundary conditions
# The actual fluid domain for this rank is f[1:local_Ny+1, :, :]

def equilibrium_pdf(rho_local_eq, ux_local_eq, uy_local_eq): # Renamed params to avoid clash
    """Calculates the equilibrium distribution function."""
    feq = np.zeros(9)
    for i_pop in range(9): # Renamed loop var
        cu = c[i_pop, 0] * ux_local_eq + c[i_pop, 1] * uy_local_eq
        feq[i_pop] = w[i_pop] * rho_local_eq * (
            1 + cu/cs2 + 0.5 * (cu/cs2)**2 - 0.5 * (ux_local_eq**2 + uy_local_eq**2)/cs2
        )
    return feq

# Initialize PDFs to equilibrium with zero velocity and unit density for the local domain
for j_local in range(1, local_Ny + 1): # Iterate over actual local fluid rows
    for i_local in range(Nx): # Iterate over all columns
        f[j_local, i_local, :] = equilibrium_pdf(rho0, 0, 0)
        rho[j_local, i_local] = rho0
        ux[j_local, i_local] = 0.0
        uy[j_local, i_local] = 0.0


# Main Loop
for t_step in range(max_timesteps):
    # Collision Step (BGK) - Applied to local domain including physical boundaries if they are local
    # The domain for collision is f[1...local_Ny, :, :]
    for j_local in range(1, local_Ny + 1):
        for i_local in range(Nx): # Loop over all columns
            feq_local = equilibrium_pdf(rho[j_local,i_local], ux[j_local,i_local], uy[j_local,i_local])
            f[j_local,i_local,:] = f[j_local,i_local,:] - (1/tau) * (f[j_local,i_local,:] - feq_local)

    # Halo Exchange (交換上下邊界的f值)
    # Send to top neighbor (rank+1), receive from bottom neighbor (rank-1) into halo f[0,:,:]
    # Send to bottom neighbor (rank-1), receive from top neighbor (rank+1) into halo f[local_Ny+1,:,:]
    
    # Non-blocking send/recv for halo exchange
    requests = []
    # Send to rank-1 (my bottom row, their top halo)
    if rank > 0:
        send_data_bottom = np.copy(f[1, :, :]) # Send first actual fluid row
        req_send_bottom = comm.Isend(send_data_bottom, dest=rank-1, tag=0)
        requests.append(req_send_bottom)
        
        recv_buffer_bottom = np.empty_like(f[0, :, :])
        req_recv_bottom = comm.Irecv(recv_buffer_bottom, source=rank-1, tag=1)
        requests.append(req_recv_bottom)

    # Send to rank+1 (my top row, their bottom halo)
    if rank < size - 1:
        send_data_top = np.copy(f[local_Ny, :, :]) # Send last actual fluid row
        req_send_top = comm.Isend(send_data_top, dest=rank+1, tag=1) # Tag must match recv tag logic
        requests.append(req_send_top)

        recv_buffer_top = np.empty_like(f[local_Ny+1, :, :])
        req_recv_top = comm.Irecv(recv_buffer_top, source=rank+1, tag=0) # Tag must match send tag logic
        requests.append(req_recv_top)

    # Wait for all non-blocking operations to complete
    if requests:
        MPI.Request.Waitall(requests)

        # Populate halos with received data
        if rank > 0:
            f[0, :, :] = recv_buffer_bottom
        if rank < size - 1:
            f[local_Ny+1, :, :] = recv_buffer_top
            
    # Streaming Step
    # f_new is local, including its own halo layers that are NOT exchanged yet.
    # Streaming will populate f_new[1...local_Ny] using f[0...local_Ny+1] (which now includes exchanged halos)
    f_new = np.zeros_like(f) 
    for j_local in range(1, local_Ny + 1): # Stream into actual local fluid rows
        for i_local in range(Nx):
            for k_pop in range(9):
                prev_y_local, prev_x_local = j_local - c[k_pop,1], i_local - c[k_pop,0]
                # prev_y_local will range from 0 to local_Ny+1 due to c[k,1]
                # This means it correctly accesses the halo regions f[0,:,:] and f[local_Ny+1,:,:]
                f_new[j_local,i_local,k_pop] = f[prev_y_local, prev_x_local, k_pop]
    
    # Boundary Conditions applied to f_new
    # No-slip walls (Left: i_local=0, Right: i_local=Nx-1) - Bounce-back for all processes on their local part
    for j_local in range(1, local_Ny + 1):
        # Left wall (i_local=0)
        f_new[j_local, 0, 1] = f_new[j_local, 0, 3]
        f_new[j_local, 0, 5] = f_new[j_local, 0, 7]
        f_new[j_local, 0, 8] = f_new[j_local, 0, 6]
        # Right wall (i_local=Nx-1)
        f_new[j_local, Nx-1, 3] = f_new[j_local, Nx-1, 1]
        f_new[j_local, Nx-1, 7] = f_new[j_local, Nx-1, 5]
        f_new[j_local, Nx-1, 6] = f_new[j_local, Nx-1, 8]

    # Bottom wall (global j=0) - Bounce-back only by rank 0
    if rank == 0:
        # Applied to the first actual fluid row: f_new[1, i_local, k]
        for i_local in range(Nx):
            f_new[1, i_local, 2] = f_new[1, i_local, 4]  # N from S
            f_new[1, i_local, 5] = f_new[1, i_local, 7]  # NE from SW
            f_new[1, i_local, 6] = f_new[1, i_local, 8]  # NW from SE

    # Moving Lid (global j=Ny_global-1) - Equilibrium-based only by the last rank
    if start_row_global + local_Ny == Ny_global: # This rank owns the global top boundary
        # Applied to the last actual fluid row: f_new[local_Ny, i_local, k]
        for i_local in range(Nx):
            # Use macroscopic rho from previous step at the lid for stability
            # rho_for_lid_feq is rho[local_Ny, i_local] (already computed for this row)
            feq_lid_unknowns = equilibrium_pdf(rho[local_Ny,i_local], u_lid, 0.0)
            f_new[local_Ny, i_local, 4] = feq_lid_unknowns[4] # S
            f_new[local_Ny, i_local, 7] = feq_lid_unknowns[7] # SW
            f_new[local_Ny, i_local, 8] = feq_lid_unknowns[8] # SE
    
    f = f_new # Update f with the post-streaming and post-BC values

    # Macroscopic Variables (recalculate for local domain including physical boundaries)
    # Rho, ux, uy are calculated for j_local in [1, local_Ny]
    for j_local in range(1, local_Ny + 1):
        for i_local in range(Nx):
            rho[j_local,i_local] = np.sum(f[j_local,i_local,:])
            ux_val = 0.0
            uy_val = 0.0
            for k_pop in range(9):
                ux_val += f[j_local,i_local,k_pop] * c[k_pop,0]
                uy_val += f[j_local,i_local,k_pop] * c[k_pop,1]
            ux[j_local,i_local] = ux_val / rho[j_local,i_local]
            uy[j_local,i_local] = uy_val / rho[j_local,i_local]

    # Global reduction for average density (for monitoring)
    if t_step % 100 == 0:
        local_rho_sum = np.sum(rho[1:local_Ny+1, :]) # Summing only active fluid cells
        global_rho_sum = comm.allreduce(local_rho_sum, op=MPI.SUM)
        avg_global_rho = global_rho_sum / (Nx_global * Ny_global)
        if rank == 0:
            print(f"Time step: {t_step}, Avg. Global Density: {avg_global_rho:.6f}")
            # To verify, print max velocities from rank 0's local part (not global max)
            # print(f"Rank 0 Max ux: {np.max(np.abs(ux[1:local_Ny+1, :]))}, Max uy: {np.max(np.abs(uy[1:local_Ny+1, :]))}")


# Gather results (e.g., ux, uy) to rank 0 for saving or plotting
local_ux_data = ux[1:local_Ny+1, :].copy() # Ensure it's contiguous and correct slice
local_uy_data = uy[1:local_Ny+1, :].copy()

# Each process sends its local_Ny * Nx (the number of elements in its local_ux_data)
num_elements_local_ux = local_ux_data.size
num_elements_local_uy = local_uy_data.size

# Gather the number of elements from each process at root
counts_ux = comm.gather(num_elements_local_ux, root=0)
counts_uy = comm.gather(num_elements_local_uy, root=0) # Should be same as counts_ux

gathered_ux_flat = None
gathered_uy_flat = None
displacements_ux = None
displacements_uy = None # Should be same as displacements_ux

if rank == 0:
    # Prepare flat receive buffers on rank 0
    gathered_ux_flat = np.empty(Nx_global * Ny_global, dtype=float)
    gathered_uy_flat = np.empty(Nx_global * Ny_global, dtype=float)
    
    # Convert counts to numpy arrays for cumsum if they aren't already (gather returns list)
    counts_ux_np = np.array(counts_ux)
    counts_uy_np = np.array(counts_uy)

    # Calculate displacements for Gatherv
    # displacements are the starting index in the flat receive buffer for each process's data
    displacements_ux = np.insert(np.cumsum(counts_ux_np[:-1]), 0, 0) if size > 0 else np.array([0])
    displacements_uy = np.insert(np.cumsum(counts_uy_np[:-1]), 0, 0) if size > 0 else np.array([0])


# Using Gatherv to collect data from all processes to rank 0
# Processes send their flattened local data. Rank 0 receives into a flattened global array.
comm.Gatherv(sendbuf=local_ux_data.flatten(), 
             recvbuf=[gathered_ux_flat, counts_ux, displacements_ux, MPI.DOUBLE], 
             root=0)
comm.Gatherv(sendbuf=local_uy_data.flatten(), 
             recvbuf=[gathered_uy_flat, counts_uy, displacements_uy, MPI.DOUBLE], 
             root=0)


if rank == 0:
    # Reshape the flat arrays to 2D global arrays on rank 0
    gathered_ux = gathered_ux_flat.reshape(Ny_global, Nx_global)
    gathered_uy = gathered_uy_flat.reshape(Ny_global, Nx_global)

    print("Simulation Finished.")
    print(f"Kinematic viscosity (nu) from tau: {nu}")
    reynolds_number = u_lid * Nx_global / nu if nu > 0 else float('inf')
    print(f"Lid Reynolds number (Re = u_lid * Nx_global / nu): {reynolds_number}")

    # Post-processing: Extract and save velocity profiles
    # Vertical centerline u-velocity (at x = (Nx_global-1)//2)
    x_center_idx = (Nx_global - 1) // 2
    u_profile_vertical = gathered_ux[:, x_center_idx]
    y_coords_normalized = np.linspace(0, 1, Ny_global)
    
    # Save u_profile_vertical
    u_profile_data = np.vstack((y_coords_normalized, u_profile_vertical)).T # Transpose to get N_rows x 2 columns
    np.savetxt('u_profile_vertical_centerline.txt', u_profile_data, 
               fmt='%.6e', header='y_normalized u_velocity', comments='')

    # Horizontal centerline v-velocity (at y = (Ny_global-1)//2)
    y_center_idx = (Ny_global - 1) // 2
    v_profile_horizontal = gathered_uy[y_center_idx, :]
    x_coords_normalized = np.linspace(0, 1, Nx_global)

    # Save v_profile_horizontal
    v_profile_data = np.vstack((x_coords_normalized, v_profile_horizontal)).T # Transpose
    np.savetxt('v_profile_horizontal_centerline.txt', v_profile_data,
               fmt='%.6e', header='x_normalized v_velocity', comments='')
    
    if rank == 0: # Ensure this print is also rank 0 only
        print("Velocity profiles saved to .txt files.")

    # Optional: Save results by rank 0 (original npz save)
    # np.savez(f"cavity_flow_mpi_results_size{size}.npz", ux=gathered_ux, uy=gathered_uy)
    
    # For plotting (example):
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8,8))
    # velocity_magnitude = np.sqrt(gathered_ux**2 + gathered_uy**2)
    # plt.imshow(velocity_magnitude.T, origin='lower', cmap='jet') # Transpose for X-Y convention
    # plt.colorbar(label='Velocity Magnitude')
    # plt.title(f'MPI Velocity Magnitude at t={max_timesteps} (size={size})')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.savefig(f"cavity_mpi_velo_size{size}.png")
    # plt.show()

# MPI Finalization
MPI.Finalize()

# if __name__ == '__main__': # This is not typically used with mpi4py scripts run via mpirun
#    pass
