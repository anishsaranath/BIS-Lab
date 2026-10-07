import numpy as np

def pso_cloud_allocation(
    vm_cpu,
    vm_ram,
    server_cpu_cap,
    server_ram_cap,
    idle_power,
    max_power,
    num_particles=10,
    max_iter=30,
    w=0.5,
    c1=1.0,
    c2=1.2,
    penalty_weight=1e5
):
    num_vms = len(vm_cpu)
    num_servers = len(server_cpu_cap)

    particles = np.random.uniform(0, num_servers, size=(num_particles, num_vms))
    velocities = np.random.uniform(-1, 1, size=(num_particles, num_vms))

    pbest = np.copy(particles)
    pbest_fitness = np.full(num_particles, np.inf)

    gbest = None
    gbest_fitness = np.inf

    def fitness_function(position):
        allocation = np.clip(np.floor(position).astype(int), 0, num_servers - 1)
        
        server_cpu_used = np.zeros(num_servers)
        server_ram_used = np.zeros(num_servers)

        for vm_idx, server_idx in enumerate(allocation):
            server_cpu_used[server_idx] += vm_cpu[vm_idx]
            server_ram_used[server_idx] += vm_ram[vm_idx]

        cpu_overload = np.maximum(0, server_cpu_used - server_cpu_cap)
        ram_overload = np.maximum(0, server_ram_used - server_ram_cap)
        penalty = penalty_weight * (np.sum(cpu_overload) + np.sum(ram_overload))

        total_energy = 0.0
        resource_imbalance = 0.0

        for s in range(num_servers):
            if server_cpu_used[s] > 0 or server_ram_used[s] > 0:
                cpu_util = server_cpu_used[s] / server_cpu_cap[s]
                ram_util = server_ram_used[s] / server_ram_cap[s]

                power = idle_power[s] + (max_power[s] - idle_power[s]) * cpu_util
                total_energy += power

                imbalance = abs(cpu_util - ram_util)
                resource_imbalance += imbalance

        return total_energy + resource_imbalance + penalty, total_energy

    for _ in range(max_iter):
        for i in range(num_particles):
            current_fitness, _ = fitness_function(particles[i])

            if current_fitness < pbest_fitness[i]:
                pbest_fitness[i] = current_fitness
                pbest[i] = np.copy(particles[i])

            if current_fitness < gbest_fitness:
                gbest_fitness = current_fitness
                gbest = np.copy(particles[i])

        for i in range(num_particles):
            r1 = np.random.rand(num_vms)
            r2 = np.random.rand(num_vms)

            cognitive = c1 * r1 * (pbest[i] - particles[i])
            social = c2 * r2 * (gbest - particles[i])
            velocities[i] = w * velocities[i] + cognitive + social

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, num_servers - 0.001)

    best_allocation = np.clip(np.floor(gbest).astype(int), 0, num_servers - 1)
    _, estimated_energy = fitness_function(gbest)

    return best_allocation, estimated_energy

if __name__ == "__main__":
    vm_cpu = np.array([2, 4, 1, 8, 2, 4, 2, 1])
    vm_ram = np.array([4, 8, 2, 16, 4, 8, 4, 2])

    server_cpu_cap = np.array([16, 16, 32])
    server_ram_cap = np.array([32, 32, 64])
    idle_power = np.array([100, 100, 150])
    max_power = np.array([250, 250, 400])

    allocation, energy = pso_cloud_allocation(
        vm_cpu, vm_ram, server_cpu_cap, server_ram_cap, idle_power, max_power
    )

    print("Optimal Allocation (VM -> Server):", allocation)
    print("Estimated Energy Consumption:", energy)
