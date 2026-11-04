import math
import random

def levy_flight(Lambda):
    num = math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2)
    den = math.gamma((1 + Lambda) / 2) * Lambda * (2 ** ((Lambda - 1) / 2))
    sigma = (num / den) ** (1 / Lambda)
    
    u = random.normalvariate(0, sigma)
    v = random.normalvariate(0, 1)
    step = u / (abs(v) ** (1 / Lambda))
    return step

class CuckooSearchScheduler:
    def __init__(self, subjects, student_conflicts, room_capacities, subject_sizes, num_slots, num_rooms, n_cuckoos=25, pa=0.25):
        self.subjects = subjects
        self.conflicts = student_conflicts
        self.room_caps = room_capacities
        self.sub_sizes = subject_sizes
        self.num_slots = num_slots
        self.num_rooms = num_rooms
        self.n_cuckoos = n_cuckoos
        self.pa = pa
        self.num_subs = len(subjects)

    def init_population(self):
        pop = []
        for _ in range(self.n_cuckoos):
            nest = []
            for _ in range(self.num_subs):
                slot = random.randint(0, self.num_slots - 1)
                room = random.randint(0, self.num_rooms - 1)
                nest.extend([slot, room])
            pop.append(nest)
        return pop

    def fitness(self, nest):
        penalty = 0
        schedule = []
        for i in range(self.num_subs):
            slot = nest[2 * i]
            room = nest[2 * i + 1]
            schedule.append((slot, room))
            if self.sub_sizes[i] > self.room_caps[room]:
                penalty += (self.sub_sizes[i] - self.room_caps[room]) * 10
        
        for i in range(self.num_subs):
            for j in range(i + 1, self.num_subs):
                if schedule[i][0] == schedule[j][0]:
                    if schedule[i][1] == schedule[j][1]:
                        penalty += 1000
                    if self.conflicts[i][j] > 0:
                        penalty += self.conflicts[i][j] * 100
        return penalty

    def clip_nest(self, nest):
        clipped = []
        for i in range(self.num_subs):
            slot = int(round(nest[2 * i]))
            room = int(round(nest[2 * i + 1]))
            slot = max(0, min(slot, self.num_slots - 1))
            room = max(0, min(room, self.num_rooms - 1))
            clipped.extend([slot, room])
        return clipped

    def solve(self, max_iter=100):
        population = self.init_population()
        fitnesses = [self.fitness(n) for n in population]
        
        best_fit = min(fitnesses)
        best_idx = fitnesses.index(best_fit)
        best_nest = list(population[best_idx])

        for _ in range(max_iter):
            for i in range(self.n_cuckoos):
                new_nest = []
                for d in range(len(population[i])):
                    step = levy_flight(1.5)
                    val = population[i][d] + step * (population[i][d] - best_nest[d])
                    new_nest.append(val)
                
                new_nest = self.clip_nest(new_nest)
                new_fit = self.fitness(new_nest)
                
                random_idx = random.randint(0, self.n_cuckoos - 1)
                if new_fit < fitnesses[random_idx]:
                    population[random_idx] = new_nest
                    fitnesses[random_idx] = new_fit

            for i in range(self.n_cuckoos):
                if random.random() < self.pa:
                    j, k = random.sample(range(self.n_cuckoos), 2)
                    abandoned_nest = []
                    for d in range(len(population[i])):
                        step = random.random() * (population[j][d] - population[k][d])
                        abandoned_nest.append(population[i][d] + step)
                    
                    abandoned_nest = self.clip_nest(abandoned_nest)
                    abandoned_fit = self.fitness(abandoned_nest)
                    if abandoned_fit < fitnesses[i]:
                        population[i] = abandoned_nest
                        fitnesses[i] = abandoned_fit

            current_min_fit = min(fitnesses)
            if current_min_fit < best_fit:
                best_fit = current_min_fit
                best_idx = fitnesses.index(best_fit)
                best_nest = list(population[best_idx])

        return best_nest, best_fit

if __name__ == "__main__":
    subjects = list(range(5))
    
    conflicts = [
        [0, 2, 0, 1, 0],
        [2, 0, 3, 0, 1],
        [0, 3, 0, 0, 0],
        [1, 0, 0, 0, 6],
        [0, 1, 0, 6, 0]
    ]
    
    room_caps = [30, 60, 45]
    sub_sizes = [25, 55, 40, 20, 50]
    slots = 6
    rooms = 3

    cs = CuckooSearchScheduler(subjects, conflicts, room_caps, sub_sizes, slots, rooms)
    best_schedule, score = cs.solve(max_iter=200)
    
    print("Best Nest Vector:", best_schedule)
    print("Penalty Score:", score)
    print("\n--- Final Exam Timetable ---")
    for sub in subjects:
        slot_assigned = best_schedule[2 * sub]
        room_assigned = best_schedule[2 * sub + 1]
        print(f"Subject {sub} -> Time Slot: {slot_assigned}, Room: {room_assigned} (Size: {sub_sizes[sub]}/Cap: {room_caps[room_assigned]})")
