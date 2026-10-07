import random

COURSES = {
    'C1': 60,
    'C2': 35,
    'C3': 80,
    'C4': 45,
    'C5': 25
}
ROOMS = {
    'r1': 40,
    'r2': 60,
    'r3': 100
}

COURSE_LIST = list(COURSES.keys())
ROOM_LIST = list(ROOMS.keys())

POP_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 3

def calculate_fitness(chromosome):
    capacity_violations = 0
    classroom_clashes = 0
    
    room_counts = {room: 0 for room in ROOM_LIST}
    
    for i, course in enumerate(COURSE_LIST):
        assigned_room = chromosome[i]
        student_strength = COURSES[course]
        room_capacity = ROOMS[assigned_room]
        if student_strength > room_capacity:
            capacity_violations += (student_strength - room_capacity)
            
        room_counts[assigned_room] += 1
        
    for room, count in room_counts.items():
        if count > 1:
            classroom_clashes += (count - 1) * 50 
            
    total_penalty = capacity_violations + classroom_clashes
    return total_penalty, capacity_violations, classroom_clashes

def create_individual():
    return [random.choice(ROOM_LIST) for _ in range(len(COURSE_LIST))]

def crossover(parent1, parent2):
    point = random.randint(1, len(COURSE_LIST) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = random.choice(ROOM_LIST)
    return chromosome

def tournament_selection(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=lambda ind: calculate_fitness(ind)[0])
    return tournament[0]

def run_genetic_algorithm():
    population = [create_individual() for _ in range(POP_SIZE)]
    
    best_individual = None
    best_fitness = float('inf')
    
    for generation in range(GENERATIONS):
        population.sort(key=lambda ind: calculate_fitness(ind)[0])
        
        current_best_fitness = calculate_fitness(population[0])[0]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[0]
            
        if best_fitness == 0:
            break
            
        next_generation = population[:2] 
        
        while len(next_generation) < POP_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            
            c1, c2 = crossover(p1, p2)
            
            next_generation.append(mutate(c1))
            if len(next_generation) < POP_SIZE:
                next_generation.append(mutate(c2))
                
        population = next_generation

    total_penalty, cap_violations, clashes = calculate_fitness(best_individual)
    
    print()
    print("OPTIMAL CLASSROOM ALLOCATION FOUND")
    print()
    for i, course in enumerate(COURSE_LIST):
        room = best_individual[i]
        print(f"Course {course} (Strength: {COURSES[course]}) -> Room {room} (Capacity: {ROOMS[room]})")
        
    print()
    print(f"Total Penalty Score: {total_penalty}")
    print(f"Capacity Violations: {cap_violations} unseated students")
    print(f"Classroom Clashes  : {clashes // 50} room sharing conflicts")
    print()

    print()
    print("ANISH SARANATH")
    print("1WN24CS041")

if __name__ == "__main__":
    run_genetic_algorithm()

